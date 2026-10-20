#!/usr/bin/env python3
"""Portable launcher. --help / --audit-only do not import ML libraries."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

from release_utils import ROOT, audit_data, environment_metadata, sha256_file, write_json


def parser():
    p = argparse.ArgumentParser(description="m7G-DOLL release launcher; original training defaults are retained.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--audit-only", action="store_true", help="validate CSVs and checksums without importing model packages")
    mode.add_argument("--check-environment", action="store_true", help="check package imports and CUDA; do not load pretrained weights")
    mode.add_argument("--smoke-test", action="store_true", help="load actual models; one small forward/backward per branch; no CV/test evaluation")
    mode.add_argument("--train", action="store_true", help="run original five-fold CV, full-training retraining and final independent evaluation")
    p.add_argument("--dataset", choices=("full", "mature", "both"), default="both")
    p.add_argument("--data-dir", type=Path, default=Path(os.environ.get("M7G_DATA_DIR", str(ROOT / "dataset"))))
    p.add_argument("--output-dir", type=Path, default=Path(os.environ.get("M7G_OUTPUT_DIR", str(ROOT / "results"))))
    p.add_argument("--dnabert-local-dir", type=Path, default=Path(os.environ["DNABERT_LOCAL_DIR"]) if os.environ.get("DNABERT_LOCAL_DIR") else None)
    p.add_argument("--dnabert-repo-id", default=os.environ.get("M7G_DNABERT_REPO_ID", "zhihan1996/DNABERT-2-117M"))
    p.add_argument("--dnabert-revision", default=os.environ.get("M7G_DNABERT_REVISION"), help="original pretrained commit/tag, if confirmed")
    p.add_argument("--dnabert-code-revision", default=os.environ.get("M7G_DNABERT_CODE_REVISION"), help="optional separate remote-code revision")
    p.add_argument("--offline", action="store_true", help="set HF offline mode; RNA-FM must also already be cached")
    p.add_argument("--require-cuda", action="store_true", help="refuse smoke/training if CUDA is unavailable")
    p.add_argument("--allow-custom-data", action="store_true", help="allow intentional checksum changes; retain schema/length/center/overlap checks")
    p.add_argument("--report", type=Path, help="optionally save the audit/environment JSON without starting training")
    return p


def probe_environment(require_cuda=False):
    metadata = environment_metadata()
    failures = []
    for module in ("numpy", "pandas", "sklearn", "torch", "fm", "transformers", "einops"):
        try:
            __import__(module)
        except Exception as exc:
            failures.append({"module": module, "error_type": type(exc).__name__, "message": str(exc)})
    metadata["import_failures"] = failures
    try:
        import torch
        metadata["torch_version"] = torch.__version__
        metadata["torch_cuda_build"] = torch.version.cuda
        metadata["cuda_available"] = torch.cuda.is_available()
    except Exception:
        metadata["cuda_available"] = False
    metadata["status"] = "passed" if not failures and not metadata["missing_packages"] and (not require_cuda or metadata["cuda_available"]) else "failed"
    return metadata


def ensure_empty_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output directory is not empty. Choose a NEW --output-dir; existing results will not be reused or overwritten.")


def configure_runtime(args):
    # Configuration only: no change to model layers, losses, CV splits or learning rates.
    os.environ["M7G_DATA_DIR"] = str(args.data_dir)
    os.environ["M7G_OUTPUT_DIR"] = str(args.output_dir)
    os.environ["M7G_DATASET"] = args.dataset
    os.environ["M7G_DNABERT_REPO_ID"] = args.dnabert_repo_id
    os.environ["M7G_OFFLINE"] = "1" if args.offline else "0"
    os.environ["DNABERT_LOCAL_DIR"] = str(args.dnabert_local_dir) if args.dnabert_local_dir else ""
    for key, value in (("M7G_DNABERT_REVISION", args.dnabert_revision), ("M7G_DNABERT_CODE_REVISION", args.dnabert_code_revision)):
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def smoke_test(runtime):
    import torch
    runtime["setup_seed"](runtime["SEED"])
    rnafm, batch_converter = runtime["load_rnafm"]()
    tokenizer, dnabert = runtime["load_dnabert"]()
    records = []
    for cfg in runtime["DATASETS"]:
        seqs, labels = runtime["load_sequences"](cfg["train_csv"])
        # Training samples only. No independent-test predictions are generated.
        indices = [next(i for i, label in enumerate(labels) if label == cls) for cls in (0, 1)]
        small_seqs, small_labels = [seqs[i] for i in indices], labels[indices]
        branches = (
            ("RNA-FM", runtime["RNAFMClassifier"](rnafm), "rna", runtime["make_rnafm_collate"](batch_converter)),
            ("DNABERT-2", runtime["DNABERT2Classifier"](dnabert), "dna", runtime["make_dnabert_collate"](tokenizer)),
        )
        for tag, model, alphabet, collate in branches:
            model.to(runtime["DEVICE"])
            model.train()
            model.zero_grad(set_to_none=True)
            loader = runtime["make_loader"](small_seqs, small_labels, collate, alphabet, batch_size=2)
            inputs, y = next(iter(loader))
            logits = model(*[v.to(runtime["DEVICE"]) for v in inputs])
            if tuple(logits.shape) != (2, 2) or not torch.isfinite(logits).all():
                raise RuntimeError(tag + ": invalid smoke-test logits")
            weights, _ = runtime["build_class_weights"](small_labels)
            loss = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=runtime["LABEL_SMOOTHING"])(logits, y.to(runtime["DEVICE"]))
            if not torch.isfinite(loss):
                raise RuntimeError(tag + ": non-finite smoke-test loss")
            loss.backward()
            trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
            if not any("lora_B" in n and p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum().item() > 0 for n, p in trainable.items()):
                raise RuntimeError(tag + ": no nonzero finite LoRA-B gradient; inspect the actual pretrained forward path")
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError(tag + ": frozen parameters unexpectedly have gradients")
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in trainable.values()):
                raise RuntimeError(tag + ": non-finite trainable gradients")
            records.append({"dataset": cfg["name"], "branch": tag, "logits_shape": list(logits.shape), "loss": float(loss.item()), "lora_gradient_check": "passed"})
            model.zero_grad(set_to_none=True)
            print("SMOKE PASSED: {} / {}".format(cfg["name"], tag))
    return {"status": "passed", "scope": "Actual model forward/backward on two training rows per branch and dataset only. No optimizer step, CV, checkpoint selection, independent-test evaluation, or published-metric reproduction.", "checks": records}


def main(argv=None):
    args = parser().parse_args(argv)
    args.data_dir, args.output_dir = args.data_dir.expanduser().resolve(), args.output_dir.expanduser().resolve()
    if args.dnabert_local_dir:
        args.dnabert_local_dir = args.dnabert_local_dir.expanduser().resolve()
        if not args.dnabert_local_dir.is_dir():
            raise ValueError("Configured --dnabert-local-dir does not exist")
    if args.dnabert_local_dir and (args.dnabert_revision or args.dnabert_code_revision):
        raise ValueError("Use either a local snapshot or remote revision flags, not both; record the local snapshot identity separately")
    if args.check_environment:
        report = probe_environment(args.require_cuda)
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    report = audit_data(args.data_dir, args.dataset, args.allow_custom_data)
    if args.audit_only:
        if args.report:
            write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    ensure_empty_output(args.output_dir)
    configure_runtime(args)
    environment = probe_environment(args.require_cuda)
    if environment["status"] != "passed":
        print(json.dumps(environment, ensure_ascii=False, indent=2))
        raise ValueError("Environment check failed; no model weights were loaded and no training started")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_identity = None
    if args.dnabert_local_dir:
        local_identity = {str(p.relative_to(args.dnabert_local_dir)): sha256_file(p)
                          for p in sorted(args.dnabert_local_dir.rglob("*"))
                          if p.is_file() and p.suffix in (".json", ".py", ".txt") and ".git" not in p.parts}
    manifest = {"status": "started", "mode": "smoke-test" if args.smoke_test else "train", "start_unix": time.time(),
                "data_audit": report, "environment": environment, "model_script_sha256": sha256_file(ROOT / "dual_plm.py"),
                "dnabert_source": str(args.dnabert_local_dir) if args.dnabert_local_dir else args.dnabert_repo_id,
                "dnabert_revision": args.dnabert_revision, "dnabert_code_revision": args.dnabert_code_revision,
                "local_snapshot_code_config_hashes": local_identity,
                "pretrained_weight_identity_warning": "RNA-FM weight hash and exact historical pretrained revisions are not supplied; local snapshot hashes cover code/config/text, NOT binary weights. Confirm before a reproducibility claim.",
                "cached_results_reused": False}
    write_json(args.output_dir / "run_manifest.json", manifest)
    try:
        runtime = runpy.run_path(str(ROOT / "dual_plm.py"), run_name="m7g_doll_runtime")
        if args.smoke_test:
            result = smoke_test(runtime)
            write_json(args.output_dir / "smoke_test.json", result)
        else:
            runtime["main"]()
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["end_unix"] = time.time()
        write_json(args.output_dir / "run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        sys.exit(2)
