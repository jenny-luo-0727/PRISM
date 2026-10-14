"""Train a PRISM reaction regressor: reaction text plus per-component molecule graphs, optional LoRA on the LLM.

Follows ``src.materials.prism_train`` (smooth-L1 on train-standardized targets, AdamW, per-epoch cosine
schedule, gradient-norm clip 5.0, fixed epochs with the best-validation-MAE epoch selected), with four
differences: splits come from the exported ``reactions.csv`` instead of an ID hash; each reaction's
component molecules are encoded separately (``src.reactions.model``); ``--lora`` adds torchtune LoRA to
the frozen LLM; and ``--seed`` seeds only training (initialization and batch order), never the split.

Precision: the backbone loads in bf16. ``apply_lora_torchtune`` rebuilds every adapted layer in fp32,
which a bf16 forward rejects, so the frozen base weights are cast back to bf16 while the LoRA matrices,
graph encoder, projector and head keep fp32 parameters. Every forward runs under bf16 autocast.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import ModelConfig
from src.materials.data import HuggingFaceTokenizer
from src.model import UnifiedTransformer
from src.utils.lora_utils import apply_lora_torchtune

from .data import ReactionDataset, collate_reactions, load_reactions, molecule_graph, shuffled_molecule_map
from .model import MODES, PRISMReactionRegressor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", required=True, help="column of reactions.csv holding train/val/test/unused")
    parser.add_argument("--components", default="thiol_key,imine_key,catalyst_id",
                        help="comma-separated id columns, in graph-token order")
    parser.add_argument("--mode", choices=MODES, default="joint")
    parser.add_argument("--text-model", required=True, help="HF id or local snapshot directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="training seed (init + batch order), not the split")
    parser.add_argument("--shuffle-seed", type=int, default=0, help="derangement seed for --mode joint_shuffled")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="graph encoder, projector, head")
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-layers", type=int, default=3)
    parser.add_argument("--graph-tokens", type=int, default=8, help="tokens per component molecule")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--max-neighbors", type=int, default=16)
    parser.add_argument("--max-text-length", type=int, default=512, help="a longer text is an error, never truncated")
    parser.add_argument("--lora", action="store_true", help="add trainable LoRA to the frozen LLM")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=float, default=128.0, help="torchtune scales by alpha / r")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs: keep the first N reactions per partition")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = prediction - target
    denominator = float(((target - target.mean()) ** 2).sum())
    return {
        "spearman": float(pd.Series(prediction).rank().corr(pd.Series(target).rank())),
        "rmse": float(np.sqrt((error**2).mean())),
        "mae": float(np.abs(error).mean()),
        "r2": 1.0 - float((error**2).sum()) / denominator if denominator > 0 else float("nan"),
    }


def add_lora(backbone: torch.nn.Module, args: argparse.Namespace) -> None:
    """Add LoRA to every nn.Linear except lm_head; return the frozen base weights to the backbone dtype."""
    from torchtune.modules.peft import LoRALinear

    base_dtype = backbone.get_input_embeddings().weight.dtype
    apply_lora_torchtune(backbone, rank=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    for module in backbone.modules():
        if isinstance(module, LoRALinear):
            module.weight.data = module.weight.data.to(base_dtype)
            if module.use_bias:
                module.bias.data = module.bias.data.to(base_dtype)


@torch.no_grad()
def evaluate(model, loader, device, mode, target_mean, target_std):
    model.eval()
    ids, predictions, targets = [], [], []
    for batch in loader:
        ids.extend(batch["reaction_id"])
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            normalized, _ = model(batch, mode=mode)
        predictions.append(normalized.float().cpu() * target_std + target_mean)
        targets.append(batch["target"].cpu())
    prediction = torch.cat(predictions).numpy()
    target = torch.cat(targets).numpy()
    return regression_metrics(prediction, target), list(zip(ids, target.tolist(), prediction.tolist(), strict=True))


def write_predictions(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("reaction_id", "true", "predicted"))
        writer.writerows(rows)


def git_head() -> str:
    root = Path(__file__).resolve().parents[2]
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "src"], capture_output=True, text=True).stdout
    return head + ("-dirty" if dirty.strip() else "")


def run(args: argparse.Namespace) -> dict:
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    components = args.components.split(",")
    partitions, molecules = load_reactions(args.data_dir, args.split, components)
    if args.limit is not None:
        partitions = {part: records[: args.limit] for part, records in partitions.items()}
    every = [record for records in partitions.values() for record in records]
    graphs = {
        key: molecule_graph(m["atomic_numbers"], m["positions"], args.cutoff, args.max_neighbors)
        for key, m in molecules.items()
    }
    molecule_map = shuffled_molecule_map(every, args.shuffle_seed) if args.mode == "joint_shuffled" else None

    tokenizer = HuggingFaceTokenizer(str(args.text_model))
    lengths = [len(tokenizer.tokenizer.encode(record.text, add_special_tokens=True)) for record in every]
    if max(lengths) > args.max_text_length:
        raise ValueError(f"longest text is {max(lengths)} tokens > --max-text-length {args.max_text_length}")

    targets = torch.tensor([record.target for record in partitions["train"]])
    target_mean = targets.mean().item()
    target_std = targets.std(unbiased=False).clamp_min(1e-8).item()

    def loader(part: str, shuffle: bool) -> DataLoader:
        dataset = ReactionDataset(partitions[part], tokenizer, graphs, args.max_text_length, molecule_map)
        generator = torch.Generator().manual_seed(args.seed)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, generator=generator,
                          collate_fn=collate_reactions)

    train_loader, val_loader, test_loader = loader("train", True), loader("val", False), loader("test", False)

    prism = UnifiedTransformer(
        ModelConfig(
            llm_backbone_id=str(args.text_model),
            llm_tokenizer_id=str(args.text_model),
            freeze_backbone=True,
            freeze_encoders=False,
            modalities=["text"],
            attn_implementation="sdpa",
        )
    )
    prism.tokenizer = tokenizer.tokenizer
    model = PRISMReactionRegressor(
        prism,
        n_components=len(components),
        graph_hidden_dim=args.hidden_dim,
        graph_layers=args.graph_layers,
        cutoff=args.cutoff,
        graph_tokens=args.graph_tokens,
    )
    if args.lora:
        add_lora(model.prism.backbone, args)
    if args.mode == "text":
        for module in (model.prism.encoders["graph"], model.prism.projectors["graph"], model.component_embedding):
            module.requires_grad_(False)
    model.to(device)

    lora_parameters = [p for n, p in model.prism.backbone.named_parameters() if p.requires_grad]
    if any("lora_" not in n for n, p in model.prism.backbone.named_parameters() if p.requires_grad):
        raise RuntimeError("a non-LoRA backbone parameter is trainable")
    lora_ids = {id(p) for p in lora_parameters}
    task_parameters = [p for p in model.parameters() if p.requires_grad and id(p) not in lora_ids]
    groups = [{"params": task_parameters, "lr": args.learning_rate}]
    if lora_parameters:
        groups.append({"params": lora_parameters, "lr": args.lora_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}

    print(f"split={args.split} mode={args.mode} lora={args.lora} seed={args.seed} device={device} "
          f"sizes={ {k: len(v) for k, v in partitions.items()} } max_text_tokens={max(lengths)} "
          f"trainable: lora={sum(p.numel() for p in lora_parameters):,} task={sum(p.numel() for p in task_parameters):,}",
          flush=True)

    best_mae, best_epoch, best_state = math.inf, None, None
    history = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            normalized_target = (batch["target"] - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                _, loss = model(batch, mode=args.mode, targets=normalized_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            running += loss.item() * normalized_target.numel()
            seen += normalized_target.numel()
        scheduler.step()
        val_metrics, _ = evaluate(model, val_loader, device, args.mode, target_mean, target_std)
        history.append({"epoch": epoch, "train_loss": running / seen, "seconds": time.perf_counter() - epoch_start,
                        **{f"val_{k}": v for k, v in val_metrics.items()}})
        print(f"epoch={epoch:03d} loss={running / seen:.5f} sec={history[-1]['seconds']:.1f} " + " ".join(f"val_{k}={v:.4f}" for k, v in val_metrics.items()),
              flush=True)
        if val_metrics["mae"] < best_mae:
            best_mae, best_epoch = val_metrics["mae"], epoch
            best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items() if n in trainable_names}

    torch.save(best_state, args.output_dir / "best_trainable.pt")
    model.load_state_dict(best_state, strict=False)
    val_metrics, val_rows = evaluate(model, val_loader, device, args.mode, target_mean, target_std)
    test_metrics, test_rows = evaluate(model, test_loader, device, args.mode, target_mean, target_std)
    write_predictions(args.output_dir / "val_predictions.csv", val_rows)
    write_predictions(args.output_dir / "test_predictions.csv", test_rows)
    print("test " + " ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()), flush=True)

    import torchtune
    import transformers

    record = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "prism_git": git_head(),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "torchtune": torchtune.__version__},
        "data_manifest": json.loads((args.data_dir / "manifest.json").read_text()),
        "split_sizes": {k: len(v) for k, v in partitions.items()},
        "max_text_tokens": max(lengths),
        "trainable_parameters": {"lora": sum(p.numel() for p in lora_parameters),
                                 "task": sum(p.numel() for p in task_parameters)},
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": best_epoch,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    if molecule_map is not None:
        record["shuffled_molecule_map"] = {k: list(v) for k, v in molecule_map.items()}
    (args.output_dir / "run.json").write_text(json.dumps(record, indent=2))
    return test_metrics


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
