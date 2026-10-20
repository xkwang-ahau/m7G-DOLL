import os
import sys
import json
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, roc_auc_score, roc_curve
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_SMOOTHING = 0.05

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_scheduler(optimizer, total_steps, warmup_ratio=0.1):
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)
try:
    import fm
except ImportError as e:
    raise ImportError('RNA-FM is required. Install with: pip install rna-fm') from e
DNABERT_LOCAL_DIR = '/home/lchen/plm-m7g/DNABERT-2-117M'
DNABERT_REPO_ID = 'zhihan1996/DNABERT-2-117M'
if DNABERT_LOCAL_DIR and os.path.isdir(DNABERT_LOCAL_DIR):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    DNABERT_NAME = DNABERT_LOCAL_DIR
    DNABERT_SOURCE_MODE = 'local (offline)'
else:
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    DNABERT_NAME = DNABERT_REPO_ID
    DNABERT_SOURCE_MODE = f"remote via {os.environ['HF_ENDPOINT']}"
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
SEED = 42
N_FOLDS = 5
CV_EPOCHS = 20
HEAD_LR = 0.0003
PLM_LR = 0.0001
WEIGHT_DECAY = 0.002
WARMUP_RATIO = 0.1
GRAD_CLIP = 5.0
NEGATIVE_CLASS_WEIGHT_BOOST = 1.0
POSITIVE_CLASS_WEIGHT_BOOST = 1.0
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
RNAFM_BATCH_SIZE = 16
RNAFM_HIDDEN = 640
RNAFM_REPR_LAYER = 12
RNAFM_LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'out_proj')
DNABERT_BATCH_SIZE = 16
DNABERT_HIDDEN = 768
DNABERT_MAX_TOKENS = 512
DNABERT_LORA_TARGETS = ('Wqkv', 'dense')
HIDDEN_DIM = 256
DROPOUT = 0.4
ENSEMBLE_W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 4)
FIXED_THRESHOLD = 0.5
DATA_DIR = '/home/lchen/plm-m7g/dataset'
OUT_DIR = os.path.join(HERE, 'dual_plm_resultslora16')
os.makedirs(OUT_DIR, exist_ok=True)
REUSE_CACHED = True
DATASETS = [{'name': 'mostdistant401', 'train_csv': os.path.join(DATA_DIR, 'train_mostdistant401.csv'), 'test_csv': os.path.join(DATA_DIR, 'test_mostdistant401.csv')}, {'name': 'mature_mostdistant201', 'train_csv': os.path.join(DATA_DIR, 'train_mature_mostdistant201.csv'), 'test_csv': os.path.join(DATA_DIR, 'test_mature_mostdistant201.csv')}]
METRIC_NAMES = ['acc', 'sn', 'sp', 'prec', 'f1', 'mcc', 'auc']

class LoRALinear(nn.Module):

    def __init__(self, base: nn.Linear, r: int=8, alpha: int=16, dropout: float=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.alpha = alpha
        self.scaling = float(alpha) / float(r)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight + self.lora_B.weight @ self.lora_A.weight * self.scaling

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling

def inject_lora(model, target_names, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT) -> int:
    for p in model.parameters():
        p.requires_grad = False
    n_replaced = 0
    for _, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_names:
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                n_replaced += 1
    return n_replaced

def reset_lora_params(model) -> None:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.kaiming_uniform_(module.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(module.lora_B.weight)

def count_params(model):
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    total = sum((p.numel() for p in model.parameters()))
    return (trainable, total)

def _vanilla_qkvpacked_attn(qkv, bias=None, causal=False, softmax_scale=None):
    orig_dtype = qkv.dtype
    qkv = qkv.float()
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.size(-1))
    scores = torch.matmul(q, k.transpose(-1, -2)) * softmax_scale
    if bias is not None:
        scores = scores + bias.float()
    if causal:
        s = scores.size(-1)
        cmask = torch.triu(torch.ones(s, s, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(cmask, float('-inf'))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    out = out.transpose(1, 2).contiguous()
    return out.to(orig_dtype)

def patch_dnabert2_flash_attention():
    patched = []
    for mod_name in list(sys.modules.keys()):
        if 'DNABERT-2' in mod_name and mod_name.endswith('bert_layers'):
            mod = sys.modules[mod_name]
            if hasattr(mod, 'flash_attn_qkvpacked_func'):
                mod.flash_attn_qkvpacked_func = _vanilla_qkvpacked_attn
                patched.append(mod_name)
    return patched

def make_head(in_dim, hidden_dim=HIDDEN_DIM, num_classes=2, dropout=DROPOUT):
    head = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
    for m in head:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return head

class RNAFMClassifier(nn.Module):

    def __init__(self, rnafm, plm_dim=RNAFM_HIDDEN, hidden_dim=HIDDEN_DIM, dropout=DROPOUT, repr_layer=RNAFM_REPR_LAYER):
        super().__init__()
        self.rnafm = rnafm
        self.repr_layer = repr_layer
        self.head = make_head(plm_dim, hidden_dim, dropout=dropout)

    def forward(self, tokens, lengths):
        results = self.rnafm(tokens, repr_layers=[self.repr_layer])
        token_repr = results['representations'][self.repr_layer]
        B, T, _ = token_repr.shape
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        valid = (pos >= 1) & (pos <= lengths.unsqueeze(1))
        mask = valid.unsqueeze(-1).to(token_repr.dtype)
        mean_feat = (token_repr * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(mean_feat)

    def trainable_param_groups(self, head_lr, plm_lr, wd):
        plm, head = ([], [])
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (plm if n.startswith('rnafm') else head).append(p)
        return [{'params': plm, 'lr': plm_lr, 'weight_decay': wd}, {'params': head, 'lr': head_lr, 'weight_decay': wd}]

def _extract_hidden(out):
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, 'last_hidden_state'):
        return out.last_hidden_state
    return out[0]

class DNABERT2Classifier(nn.Module):

    def __init__(self, dnabert, plm_dim=DNABERT_HIDDEN, hidden_dim=HIDDEN_DIM, dropout=DROPOUT):
        super().__init__()
        self.dnabert = dnabert
        self.head = make_head(plm_dim, hidden_dim, dropout=dropout)

    def forward(self, input_ids, attention_mask):
        out = self.dnabert(input_ids=input_ids, attention_mask=attention_mask)
        token_repr = _extract_hidden(out)
        valid = attention_mask.bool()
        mask = valid.unsqueeze(-1).to(token_repr.dtype)
        mean_feat = (token_repr * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.head(mean_feat)

    def trainable_param_groups(self, head_lr, plm_lr, wd):
        plm, head = ([], [])
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (plm if n.startswith('dnabert') else head).append(p)
        groups = [{'params': head, 'lr': head_lr, 'weight_decay': wd}]
        if plm:
            groups.append({'params': plm, 'lr': plm_lr, 'weight_decay': wd})
        return groups

class SequenceDataset(Dataset):

    def __init__(self, sequences, labels, alphabet='rna'):
        seqs = [str(s).upper() for s in sequences]
        if alphabet == 'rna':
            self.sequences = [s.replace('T', 'U') for s in seqs]
        elif alphabet == 'dna':
            self.sequences = [s.replace('U', 'T') for s in seqs]
        else:
            raise ValueError(alphabet)
        self.labels = np.asarray(labels).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.sequences[idx], int(self.labels[idx]))

def load_sequences(csv_path):
    df = pd.read_csv(csv_path)
    if 'Sequence' not in df.columns or 'Label' not in df.columns:
        raise ValueError("CSV must contain 'Sequence' and 'Label' columns.")
    return (df['Sequence'].astype(str).tolist(), df['Label'].astype(int).values)

def make_rnafm_collate(batch_converter):

    def _collate(batch):
        seqs, labels = zip(*batch)
        data = [(str(i), s) for i, s in enumerate(seqs)]
        _, _, tokens = batch_converter(data)
        lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
        return ([tokens, lengths], torch.tensor(labels, dtype=torch.long))
    return _collate

def make_dnabert_collate(tokenizer):

    def _collate(batch):
        seqs, labels = zip(*batch)
        enc = tokenizer(list(seqs), return_tensors='pt', padding=True, truncation=True, max_length=DNABERT_MAX_TOKENS)
        return ([enc['input_ids'], enc['attention_mask']], torch.tensor(labels, dtype=torch.long))
    return _collate

def make_loader(sequences, labels, collate_fn, alphabet, batch_size, shuffle=False, seed=SEED):
    ds = SequenceDataset(sequences, labels, alphabet=alphabet)
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, generator=generator, drop_last=False, num_workers=0, pin_memory=False)

def classification_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    out = {'acc': float(accuracy_score(labels, preds)), 'sn': float(tp / (tp + fn + 1e-12)), 'sp': float(tn / (tn + fp + 1e-12)), 'prec': float(precision_score(labels, preds, zero_division=0)), 'f1': float(f1_score(labels, preds, zero_division=0)), 'mcc': float(matthews_corrcoef(labels, preds))}
    try:
        out['auc'] = float(roc_auc_score(labels, scores))
    except ValueError:
        out['auc'] = 0.5
    try:
        out['auprc'] = float(average_precision_score(labels, scores))
    except ValueError:
        out['auprc'] = 0.0
    return out

def find_optimal_thresholds(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    valid = thresholds <= scores.max() + 1e-06
    if not valid.any():
        return {'thr_youden': 0.5, 'thr_acc': 0.5}
    fpr_v, tpr_v, thr_v = (fpr[valid], tpr[valid], thresholds[valid])
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    total = max(1, n_pos + n_neg)
    youden = tpr_v - fpr_v
    acc_arr = (tpr_v * n_pos + (1.0 - fpr_v) * n_neg) / total
    return {'thr_youden': float(thr_v[int(np.argmax(youden))]), 'thr_acc': float(thr_v[int(np.argmax(acc_arr))])}

def build_class_weights(labels):
    cc = np.bincount(labels, minlength=2)
    total = len(labels)
    w0 = total / (2.0 * max(1, cc[0])) * NEGATIVE_CLASS_WEIGHT_BOOST
    w1 = total / (2.0 * max(1, cc[1])) * POSITIVE_CLASS_WEIGHT_BOOST
    return (torch.tensor([w0, w1], dtype=torch.float32).to(DEVICE), cc)

def train_epoch(model, loader, optimizer, scheduler, ce_criterion):
    model.train()
    tot = 0.0
    n = 0
    for inputs, y in loader:
        inputs = [x.to(DEVICE) for x in inputs]
        y = y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(*inputs)
        loss = ce_criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        tot += float(loss.item())
        n += 1
    n = max(1, n)
    return tot / n

@torch.no_grad()
def evaluate(model, loader, ce_criterion):
    model.eval()
    all_scores, all_labels = ([], [])
    tot = 0.0
    n = 0
    for inputs, y in loader:
        inputs = [x.to(DEVICE) for x in inputs]
        y = y.to(DEVICE)
        logits = model(*inputs)
        tot += float(ce_criterion(logits, y).item())
        n += 1
        all_scores.append(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    m = classification_metrics(labels, scores, threshold=0.5)
    m['loss'] = tot / max(1, n)
    m['scores'] = scores
    m['labels'] = labels
    return m

def train_loop(model, train_loader, val_loader, epochs, optimizer, scheduler, ce_criterion, fold_tag=''):
    per_epoch = []
    epoch_val_scores = {}
    epoch_val_labels = {}
    for epoch in range(epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler, ce_criterion)
        if val_loader is not None:
            vm = evaluate(model, val_loader, ce_criterion)
            thr = find_optimal_thresholds(vm['labels'], vm['scores'])
            tuned = classification_metrics(vm['labels'], vm['scores'], threshold=thr['thr_acc'])
            per_epoch.append({'epoch': epoch + 1, 'train_loss': tr_loss, 'val_loss': vm['loss'], 'acc': vm['acc'], 'sn': vm['sn'], 'sp': vm['sp'], 'prec': vm['prec'], 'f1': vm['f1'], 'mcc': vm['mcc'], 'auc': vm['auc'], 'auprc': vm['auprc'], 'thr_youden': thr['thr_youden'], 'thr_acc': thr['thr_acc'], 'acc_tuned': tuned['acc'], 'mcc_tuned': tuned['mcc']})
            epoch_val_scores[epoch + 1] = vm['scores'].copy()
            epoch_val_labels[epoch + 1] = vm['labels'].copy()
            print(f"  {fold_tag}Epoch {epoch + 1:03d} | TrainLoss {tr_loss:.4f} | ValLoss {vm['loss']:.4f} | ACC {vm['acc']:.4f} | SN {vm['sn']:.4f} | SP {vm['sp']:.4f} | F1 {vm['f1']:.4f} | MCC {vm['mcc']:.4f} | AUC {vm['auc']:.4f} | thr_acc {thr['thr_acc']:.3f} | ACC_tuned {tuned['acc']:.4f}")
        else:
            print(f'  {fold_tag}Epoch {epoch + 1:03d} | TrainLoss {tr_loss:.4f} (no val)')
    return (per_epoch, epoch_val_scores, epoch_val_labels)

def summarize_fold_metrics(metrics_list, metric_names=None):
    if metric_names is None:
        metric_names = METRIC_NAMES + ['auprc']
    summary = []
    for mn in metric_names:
        vals = [float(m[mn]) for m in metrics_list if mn in m]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        summary.append({'metric': mn, 'mean': float(arr.mean()), 'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0})
    return summary

def run_plm_pipeline(plm_tag, train_seqs, train_y, test_seqs, test_y, make_model, make_loader_fn, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    required = ['oof_scores.npy', 'oof_labels.npy', 'oof_fold_ids.npy', 'test_scores.npy', 'test_labels.npy']
    required_paths = [os.path.join(out_dir, f) for f in required]
    if REUSE_CACHED and all((os.path.exists(p) for p in required_paths)):
        print(f'\n[{plm_tag}] cached results found in {out_dir}')
        print(f'[{plm_tag}] -> SKIPPING CV + retrain (set REUSE_CACHED=False to force a re-run, or delete the subdir).')
        meta = {}
        meta_path = os.path.join(out_dir, 'pipeline_meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        return {'plm': plm_tag, 'oof_scores': np.load(os.path.join(out_dir, 'oof_scores.npy')), 'oof_labels': np.load(os.path.join(out_dir, 'oof_labels.npy')), 'oof_fold_ids': np.load(os.path.join(out_dir, 'oof_fold_ids.npy')), 'test_scores': np.load(os.path.join(out_dir, 'test_scores.npy')), 'test_labels': np.load(os.path.join(out_dir, 'test_labels.npy')), 'best_epoch': int(meta.get('best_epoch', -1)), 'cv_fold_metrics': meta.get('cv_fold_metrics', []), 'cv_summary': meta.get('cv_summary', [])}
    print('\n' + '-' * 90)
    print(f'[{plm_tag}] {N_FOLDS}-fold CV')
    print('-' * 90)
    n_train = len(train_y)
    oof_scores = np.full(n_train, np.nan, dtype=np.float64)
    oof_fold_ids = np.full(n_train, -1, dtype=np.int64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_per_epoch = {}
    fold_own_best_metrics = []
    fold_epoch_scores = {}
    fold_va_indices = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), train_y)):
        print(f'\n--- [{plm_tag}] Fold {fold_idx + 1}/{N_FOLDS} ---  Train={len(tr_idx)} | Val={len(va_idx)}')
        setup_seed(SEED)
        model = make_model()
        tr_seqs = [train_seqs[i] for i in tr_idx]
        va_seqs = [train_seqs[i] for i in va_idx]
        tr_y, va_y = (train_y[tr_idx], train_y[va_idx])
        train_loader = make_loader_fn(tr_seqs, tr_y, shuffle=True, seed=SEED)
        val_loader = make_loader_fn(va_seqs, va_y, shuffle=False, seed=SEED)
        class_weights, class_counts = build_class_weights(tr_y)
        ce_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING).to(DEVICE)
        optimizer = AdamW(model.trainable_param_groups(HEAD_LR, PLM_LR, WEIGHT_DECAY))
        scheduler = get_scheduler(optimizer, total_steps=CV_EPOCHS * len(train_loader), warmup_ratio=WARMUP_RATIO)
        n_tr, n_to = count_params(model)
        print(f'  Class counts: {class_counts.tolist()} | Trainable: {n_tr:,}/{n_to:,} ({100.0 * n_tr / n_to:.2f}%)')
        per_epoch, epoch_val_scores, _ = train_loop(model, train_loader, val_loader, CV_EPOCHS, optimizer, scheduler, ce_criterion, fold_tag=f'[{plm_tag} F{fold_idx + 1}] ')
        fold_per_epoch[fold_idx] = per_epoch
        fold_epoch_scores[fold_idx] = epoch_val_scores
        fold_va_indices.append(va_idx)
        best = max(per_epoch, key=lambda r: r['auc'])
        best = dict(best)
        best['fold'] = fold_idx + 1
        fold_own_best_metrics.append(best)
        print(f"  [{plm_tag}] Fold {fold_idx + 1} own best: epoch={best['epoch']}, AUC={best['auc']:.4f}, ACC={best['acc']:.4f}, MCC={best['mcc']:.4f}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame([dict(m, fold=fi + 1) for fi, ms in fold_per_epoch.items() for m in ms]).to_csv(os.path.join(out_dir, 'cv_per_epoch_metrics.csv'), index=False)
    pd.DataFrame(fold_own_best_metrics).to_csv(os.path.join(out_dir, 'cv_fold_own_best_metrics_diagnostic.csv'), index=False)
    auc_mat = np.zeros((N_FOLDS, CV_EPOCHS))
    for fi, ms in fold_per_epoch.items():
        for m in ms:
            auc_mat[fi, m['epoch'] - 1] = m['auc']
    avg_auc = auc_mat.mean(axis=0)
    best_epoch = int(np.argmax(avg_auc) + 1)
    print(f'[{plm_tag}] global best epoch by mean Val AUC: {best_epoch} (mean AUC = {avg_auc[best_epoch - 1]:.6f})')
    cv_fold_metrics = []
    for fold_idx, va_idx in enumerate(fold_va_indices):
        scores_at_best = fold_epoch_scores[fold_idx].get(best_epoch)
        if scores_at_best is None:
            raise RuntimeError(f'[{plm_tag}] Fold {fold_idx + 1}: no val scores recorded for epoch {best_epoch}. This should not happen.')
        oof_scores[va_idx] = scores_at_best
        oof_fold_ids[va_idx] = fold_idx
        fm = classification_metrics(train_y[va_idx], scores_at_best, FIXED_THRESHOLD)
        fm.update({'fold': fold_idx + 1, 'epoch': best_epoch, 'threshold': FIXED_THRESHOLD})
        cv_fold_metrics.append(fm)
    del fold_epoch_scores
    cv_summary = summarize_fold_metrics(cv_fold_metrics)
    pd.DataFrame(cv_fold_metrics).to_csv(os.path.join(out_dir, 'cv_fold_metrics_global_best_epoch_threshold_0.5.csv'), index=False)
    pd.DataFrame(cv_summary).to_csv(os.path.join(out_dir, 'cv_summary_global_best_epoch_threshold_0.5.csv'), index=False)
    print(f'\n[{plm_tag}] CV mean ± std at global best_epoch={best_epoch}, threshold={FIXED_THRESHOLD}:')
    for r in cv_summary:
        print(f"  {r['metric'].upper():<10}: {r['mean']:.4f} ± {r['std']:.4f}")
    print(f'\n[{plm_tag}] Retrain on FULL train set for {best_epoch} epochs')
    setup_seed(SEED)
    model_final = make_model()
    full_loader = make_loader_fn(train_seqs, train_y, shuffle=True, seed=SEED)
    test_loader = make_loader_fn(test_seqs, test_y, shuffle=False, seed=SEED)
    class_weights_full, _ = build_class_weights(train_y)
    ce_final = nn.CrossEntropyLoss(weight=class_weights_full, label_smoothing=LABEL_SMOOTHING).to(DEVICE)
    optimizer_final = AdamW(model_final.trainable_param_groups(HEAD_LR, PLM_LR, WEIGHT_DECAY))
    scheduler_final = get_scheduler(optimizer_final, total_steps=best_epoch * len(full_loader), warmup_ratio=WARMUP_RATIO)
    train_loop(model_final, full_loader, None, best_epoch, optimizer_final, scheduler_final, ce_final, fold_tag=f'[{plm_tag} Retrain] ')
    test_m = evaluate(model_final, test_loader, ce_final)
    test_scores = test_m['scores']
    test_labels = test_m['labels']
    np.save(os.path.join(out_dir, 'oof_scores.npy'), oof_scores)
    np.save(os.path.join(out_dir, 'oof_labels.npy'), np.asarray(train_y))
    np.save(os.path.join(out_dir, 'oof_fold_ids.npy'), oof_fold_ids)
    np.save(os.path.join(out_dir, 'test_scores.npy'), test_scores)
    np.save(os.path.join(out_dir, 'test_labels.npy'), test_labels)
    trainable_sd = {n: p.detach().cpu() for n, p in model_final.named_parameters() if p.requires_grad}
    torch.save(trainable_sd, os.path.join(out_dir, 'lora_head_state.pth'))
    with open(os.path.join(out_dir, 'pipeline_meta.json'), 'w') as f:
        json.dump({'best_epoch': int(best_epoch), 'threshold': float(FIXED_THRESHOLD), 'cv_fold_metrics': cv_fold_metrics, 'cv_summary': cv_summary}, f, indent=2)
    del model_final
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'plm': plm_tag, 'oof_scores': oof_scores, 'oof_labels': np.asarray(train_y), 'oof_fold_ids': oof_fold_ids, 'test_scores': test_scores, 'test_labels': test_labels, 'best_epoch': best_epoch, 'cv_fold_metrics': cv_fold_metrics, 'cv_summary': cv_summary}

def ensemble_and_report(dataset_name, res_rnafm, res_dnabert, out_dir):
    oof_labels = np.asarray(res_rnafm['oof_labels']).astype(int)
    oof_r = np.asarray(res_rnafm['oof_scores'], dtype=float)
    oof_d = np.asarray(res_dnabert['oof_scores'], dtype=float)
    fold_ids_r = np.asarray(res_rnafm['oof_fold_ids']).astype(int)
    fold_ids_d = np.asarray(res_dnabert['oof_fold_ids']).astype(int)
    test_labels = np.asarray(res_rnafm['test_labels']).astype(int)
    test_r = np.asarray(res_rnafm['test_scores'], dtype=float)
    test_d = np.asarray(res_dnabert['test_scores'], dtype=float)
    if not np.array_equal(oof_labels, np.asarray(res_dnabert['oof_labels']).astype(int)):
        raise ValueError('RNA-FM and DNABERT-2 OOF labels are not aligned.')
    if not np.array_equal(fold_ids_r, fold_ids_d):
        raise ValueError('RNA-FM and DNABERT-2 fold IDs are not aligned. Use the same CV split for both PLMs, or disable/rebuild stale cached results.')
    if not np.array_equal(test_labels, np.asarray(res_dnabert['test_labels']).astype(int)):
        raise ValueError('RNA-FM and DNABERT-2 test labels are not aligned.')
    valid_oof = ~(np.isnan(oof_r) | np.isnan(oof_d) | (fold_ids_r < 0))
    if not valid_oof.all():
        print(f'[ensemble] WARNING: {np.sum(~valid_oof)} OOF samples missing; excluded from weight search and CV summary.')
    ol = oof_labels[valid_oof]
    orr = oof_r[valid_oof]
    odd = oof_d[valid_oof]
    fold_ids = fold_ids_r[valid_oof]
    best_w, best_oof_auc = (0.5, -1.0)
    w_curve = []
    for w in ENSEMBLE_W_GRID:
        ens = w * orr + (1.0 - w) * odd
        try:
            a = roc_auc_score(ol, ens)
        except ValueError:
            a = 0.5
        row = {'w_rnafm': float(w), 'w_dnabert2': float(1.0 - w), 'oof_auc': float(a)}
        w_curve.append(row)
        if a > best_oof_auc:
            best_oof_auc, best_w = (float(a), float(w))
    oof_ens = best_w * orr + (1.0 - best_w) * odd
    oof_rnafm_m = classification_metrics(ol, orr, FIXED_THRESHOLD)
    oof_dnabert_m = classification_metrics(ol, odd, FIXED_THRESHOLD)
    oof_ens_m = classification_metrics(ol, oof_ens, FIXED_THRESHOLD)
    rnafm_cv_fold = res_rnafm.get('cv_fold_metrics', [])
    dnabert_cv_fold = res_dnabert.get('cv_fold_metrics', [])
    rnafm_cv_summary = res_rnafm.get('cv_summary', summarize_fold_metrics(rnafm_cv_fold))
    dnabert_cv_summary = res_dnabert.get('cv_summary', summarize_fold_metrics(dnabert_cv_fold))
    fusion_cv_fold = []
    for fold_idx in sorted(np.unique(fold_ids)):
        mask = fold_ids == fold_idx
        fold_prob = best_w * orr[mask] + (1.0 - best_w) * odd[mask]
        fm = classification_metrics(ol[mask], fold_prob, FIXED_THRESHOLD)
        fm.update({'fold': int(fold_idx) + 1, 'threshold': FIXED_THRESHOLD, 'w_rnafm': float(best_w), 'w_dnabert2': float(1.0 - best_w)})
        fusion_cv_fold.append(fm)
    fusion_cv_summary = summarize_fold_metrics(fusion_cv_fold)
    test_ens = best_w * test_r + (1.0 - best_w) * test_d
    rnafm_test = classification_metrics(test_labels, test_r, FIXED_THRESHOLD)
    dnabert_test = classification_metrics(test_labels, test_d, FIXED_THRESHOLD)
    ens_test = classification_metrics(test_labels, test_ens, FIXED_THRESHOLD)
    print('\n' + '=' * 90)
    print(f'[{dataset_name}] MULTI-PLM ENSEMBLE')
    print('=' * 90)
    print(f'  OOF weight search by AUC: best w_rnafm={best_w:.2f} (w_dnabert2={1 - best_w:.2f}), pooled OOF ensemble AUC={best_oof_auc:.4f}')
    print(f'  Fixed decision threshold: {FIXED_THRESHOLD:.2f}')

    def _row(tag, m):
        return f"  {tag:<28} ACC {m['acc']:.4f} | SN {m['sn']:.4f} | SP {m['sp']:.4f} | PREC {m['prec']:.4f} | F1 {m['f1']:.4f} | MCC {m['mcc']:.4f} | AUC {m['auc']:.4f}"

    def _summary_row(model_name, summary):
        d = {r['metric']: r for r in summary}
        vals = []
        for mn in METRIC_NAMES:
            if mn in d:
                vals.append(f"{mn.upper()} {d[mn]['mean']:.4f}±{d[mn]['std']:.4f}")
        return f'  {model_name:<12} ' + ' | '.join(vals)
    print('\n  --- CV mean ± std at threshold=0.5 ---')
    print(_summary_row('RNA-FM', rnafm_cv_summary))
    print(_summary_row('DNABERT-2', dnabert_cv_summary))
    print(_summary_row('ENSEMBLE', fusion_cv_summary))
    print('\n  --- pooled OOF metrics at threshold=0.5 ---')
    print(_row('RNA-FM', oof_rnafm_m))
    print(_row('DNABERT-2', oof_dnabert_m))
    print(_row('ENSEMBLE', oof_ens_m))
    print('\n  --- independent test metrics at threshold=0.5 ---')
    print(_row('RNA-FM', rnafm_test))
    print(_row('DNABERT-2', dnabert_test))
    print(_row('ENSEMBLE', ens_test))
    pd.DataFrame(w_curve).to_csv(os.path.join(out_dir, 'ensemble_weight_search_by_oof_auc.csv'), index=False)
    pd.DataFrame(fusion_cv_fold).to_csv(os.path.join(out_dir, 'ensemble_cv_fold_metrics_threshold_0.5.csv'), index=False)
    pd.DataFrame(fusion_cv_summary).to_csv(os.path.join(out_dir, 'ensemble_cv_summary_mean_std_threshold_0.5.csv'), index=False)
    cv_rows = []
    for model_name, summary in [('RNA-FM', rnafm_cv_summary), ('DNABERT-2', dnabert_cv_summary), ('ENSEMBLE', fusion_cv_summary)]:
        row = {'dataset': dataset_name, 'model': model_name, 'threshold': FIXED_THRESHOLD}
        for r in summary:
            row[f"{r['metric']}_mean"] = r['mean']
            row[f"{r['metric']}_std"] = r['std']
        cv_rows.append(row)
    pd.DataFrame(cv_rows).to_csv(os.path.join(out_dir, 'paper_table_cv_mean_std.csv'), index=False)
    test_rows = []
    for model_name, m in [('RNA-FM', rnafm_test), ('DNABERT-2', dnabert_test), ('ENSEMBLE', ens_test)]:
        row = {'dataset': dataset_name, 'model': model_name, 'threshold': FIXED_THRESHOLD, 'w_rnafm': float(best_w) if model_name == 'ENSEMBLE' else np.nan, 'w_dnabert2': float(1.0 - best_w) if model_name == 'ENSEMBLE' else np.nan}
        row.update({k: float(m[k]) for k in METRIC_NAMES + ['auprc'] if k in m})
        test_rows.append(row)
    pd.DataFrame(test_rows).to_csv(os.path.join(out_dir, 'paper_table_independent_test.csv'), index=False)
    oof_rows = []
    for model_name, m in [('RNA-FM', oof_rnafm_m), ('DNABERT-2', oof_dnabert_m), ('ENSEMBLE', oof_ens_m)]:
        row = {'dataset': dataset_name, 'model': model_name, 'threshold': FIXED_THRESHOLD}
        row.update({k: float(m[k]) for k in METRIC_NAMES + ['auprc'] if k in m})
        oof_rows.append(row)
    pd.DataFrame(oof_rows).to_csv(os.path.join(out_dir, 'pooled_oof_metrics_threshold_0.5.csv'), index=False)
    summary = {'dataset': dataset_name, 'selection_metric': 'pooled_oof_auc', 'threshold': float(FIXED_THRESHOLD), 'best_w_rnafm': float(best_w), 'best_w_dnabert2': float(1.0 - best_w), 'oof_ensemble_auc': float(best_oof_auc), 'rnafm_best_epoch': int(res_rnafm['best_epoch']), 'dnabert2_best_epoch': int(res_dnabert['best_epoch']), 'cv_mean_std': {'rnafm': rnafm_cv_summary, 'dnabert2': dnabert_cv_summary, 'ensemble': fusion_cv_summary}, 'pooled_oof': {'rnafm': {k: float(oof_rnafm_m[k]) for k in METRIC_NAMES}, 'dnabert2': {k: float(oof_dnabert_m[k]) for k in METRIC_NAMES}, 'ensemble': {k: float(oof_ens_m[k]) for k in METRIC_NAMES}}, 'independent_test': {'rnafm': {k: float(rnafm_test[k]) for k in METRIC_NAMES}, 'dnabert2': {k: float(dnabert_test[k]) for k in METRIC_NAMES}, 'ensemble': {k: float(ens_test[k]) for k in METRIC_NAMES}}}
    with open(os.path.join(out_dir, 'ensemble_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary

def load_rnafm():
    print('\nLoading RNA-FM (rna_fm_t12) ...')
    rnafm, alphabet = fm.pretrained.rna_fm_t12()
    rnafm.to(DEVICE)
    batch_converter = alphabet.get_batch_converter()
    n = inject_lora(rnafm, RNAFM_LORA_TARGETS)
    if n == 0:
        raise RuntimeError('RNA-FM: no LoRA modules matched. Inspect rnafm.named_modules().')
    n_tr, n_to = count_params(rnafm)
    print(f'  RNA-FM LoRA: replaced {n} linears; trainable {n_tr:,}/{n_to:,} ({100.0 * n_tr / n_to:.3f}%)')
    return (rnafm, batch_converter)

def _load_dnabert_backbone():
    try:
        return AutoModel.from_pretrained(DNABERT_NAME, trust_remote_code=True)
    except ValueError as e:
        if 'config_class' not in str(e):
            raise
        print('  [DNABERT-2] config_class mismatch; retrying with explicit BertConfig ...')
        from transformers import BertConfig
        cfg = BertConfig.from_pretrained(DNABERT_NAME)
        return AutoModel.from_pretrained(DNABERT_NAME, config=cfg, trust_remote_code=True)

def load_dnabert():
    if not HAS_TRANSFORMERS:
        raise ImportError('DNABERT-2 needs `transformers` and `einops`. Install with: pip install transformers einops')
    print(f'\nLoading DNABERT-2 [{DNABERT_SOURCE_MODE}]')
    print(f'  source: {DNABERT_NAME}')
    tokenizer = AutoTokenizer.from_pretrained(DNABERT_NAME, trust_remote_code=True)
    dnabert = _load_dnabert_backbone()
    patched = patch_dnabert2_flash_attention()
    if patched:
        print(f'  [DNABERT-2] flash-attention patched out (vanilla PyTorch attention) in: {patched}')
    else:
        print("  [DNABERT-2] WARNING: could not locate the bert_layers module to patch flash attention. If training crashes inside triton, inspect sys.modules for the DNABERT-2 'bert_layers' module name.")
    dnabert.to(DEVICE)
    n = inject_lora(dnabert, DNABERT_LORA_TARGETS)
    if n == 0:
        for p in dnabert.parameters():
            p.requires_grad = False
        print('  DNABERT-2 LoRA: 0 modules matched -> FALLING BACK to a frozen DNABERT-2 (head-only training).')
    else:
        n_tr, n_to = count_params(dnabert)
        print(f'  DNABERT-2 LoRA: replaced {n} linears; trainable {n_tr:,}/{n_to:,} ({100.0 * n_tr / n_to:.3f}%)')
    return (tokenizer, dnabert)

def run_dataset(dataset_cfg, rnafm, batch_converter, tokenizer, dnabert):
    name = dataset_cfg['name']
    out_dir = os.path.join(OUT_DIR, name)
    rnafm_dir = os.path.join(out_dir, 'rnafm')
    dnabert_dir = os.path.join(out_dir, 'dnabert2')
    for d in (out_dir, rnafm_dir, dnabert_dir):
        os.makedirs(d, exist_ok=True)
    print('\n' + '=' * 90)
    print(f'Dataset: {name}')
    print('=' * 90)
    setup_seed(SEED)
    train_seqs, train_y = load_sequences(dataset_cfg['train_csv'])
    test_seqs, test_y = load_sequences(dataset_cfg['test_csv'])
    print(f'  Train n={len(train_seqs)} dist={np.bincount(train_y).tolist()}; Test n={len(test_seqs)} dist={np.bincount(test_y).tolist()}')
    rnafm_collate = make_rnafm_collate(batch_converter)

    def make_rnafm_model():
        reset_lora_params(rnafm)
        return RNAFMClassifier(rnafm, dropout=DROPOUT).to(DEVICE)

    def rnafm_loader(seqs, labels, shuffle, seed):
        return make_loader(seqs, labels, rnafm_collate, alphabet='rna', batch_size=RNAFM_BATCH_SIZE, shuffle=shuffle, seed=seed)
    res_rnafm = run_plm_pipeline('RNA-FM', train_seqs, train_y, test_seqs, test_y, make_rnafm_model, rnafm_loader, rnafm_dir)
    dnabert_collate = make_dnabert_collate(tokenizer)

    def make_dnabert_model():
        reset_lora_params(dnabert)
        return DNABERT2Classifier(dnabert, dropout=DROPOUT).to(DEVICE)

    def dnabert_loader(seqs, labels, shuffle, seed):
        return make_loader(seqs, labels, dnabert_collate, alphabet='dna', batch_size=DNABERT_BATCH_SIZE, shuffle=shuffle, seed=seed)
    res_dnabert = run_plm_pipeline('DNABERT-2', train_seqs, train_y, test_seqs, test_y, make_dnabert_model, dnabert_loader, dnabert_dir)
    return ensemble_and_report(name, res_rnafm, res_dnabert, out_dir)

def main():
    print(f'Using device: {DEVICE}')
    print(f'SEED={SEED}, N_FOLDS={N_FOLDS}, CV_EPOCHS={CV_EPOCHS}')
    print(f'RNA-FM batch={RNAFM_BATCH_SIZE}; DNABERT-2 batch={DNABERT_BATCH_SIZE}')
    print(f'LoRA: r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}')
    rnafm, batch_converter = load_rnafm()
    tokenizer, dnabert = load_dnabert()
    all_summaries = []
    for cfg in DATASETS:
        if not (os.path.exists(cfg['train_csv']) and os.path.exists(cfg['test_csv'])):
            print(f"\n[Skip] {cfg['name']} — files not found.")
            continue
        all_summaries.append(run_dataset(cfg, rnafm, batch_converter, tokenizer, dnabert))
    print('\n' + '=' * 90)
    print('FINAL SUMMARY (multi-PLM ensemble)')
    print('=' * 90)
    for s in all_summaries:
        print(f"\n[{s['dataset']}]  w_rnafm={s['best_w_rnafm']:.2f} / w_dnabert2={s['best_w_dnabert2']:.2f}  (OOF ens AUC {s['oof_ensemble_auc']:.4f})")
        test_res = s['independent_test']
        print('  RNA-FM    test @0.5 :  ' + '  '.join((f"{k.upper()} {test_res['rnafm'][k]:.4f}" for k in METRIC_NAMES)))
        print('  DNABERT-2 test @0.5 :  ' + '  '.join((f"{k.upper()} {test_res['dnabert2'][k]:.4f}" for k in METRIC_NAMES)))
        print('  ENSEMBLE  test @0.5 :  ' + '  '.join((f"{k.upper()} {test_res['ensemble'][k]:.4f}" for k in METRIC_NAMES)))
    with open(os.path.join(OUT_DIR, 'all_results.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)
    print(f'\nAll outputs saved under: {OUT_DIR}')
if __name__ == '__main__':
    main()
