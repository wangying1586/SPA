from __future__ import annotations

import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import json
import math
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn

from method.pafa import PAFALoss, ProjectionHead
from models import get_backbone_class
from spa import SPAHead, SPALoss
from training.checkpoint import (
    append_jsonl,
    initialize_from_pafa_checkpoint,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
from training.meters import AverageMeter
from training.metrics import compute_classification_metrics
from util.augmentation import SpecAugment
from util.icbhi_dataset import ICBHIDataset
from util.misc import adjust_learning_rate, set_optimizer, warmup_learning_rate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "PAFA + Spherical Procrustes Alignment on ICBHI"
    )

    general = parser.add_argument_group("general")
    general.add_argument("--seed", type=int, default=1)
    general.add_argument("--print_freq", type=int, default=20)
    general.add_argument("--save_freq", type=int, default=100)
    general.add_argument("--save_dir", type=str, default="./save_spa")
    general.add_argument("--tag", type=str, default="")
    general.add_argument("--resume", type=str, default=None)
    general.add_argument("--eval", action="store_true")
    general.add_argument("--checkpoint", type=str, default=None)
    general.add_argument("--init_pafa_checkpoint", type=str, default=None)
    general.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    general.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    general.add_argument("--keep_legacy_cache", action="store_true")

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--optimizer", type=str, default="adam")
    optimization.add_argument("--epochs", type=int, default=100)
    optimization.add_argument("--learning_rate", type=float, default=5e-5)
    optimization.add_argument("--lr_decay_epochs", type=str, default="60,80")
    optimization.add_argument("--lr_decay_rate", type=float, default=0.1)
    optimization.add_argument("--weight_decay", type=float, default=1e-6)
    optimization.add_argument("--momentum", type=float, default=0.9)
    optimization.add_argument("--cosine", action="store_true")
    optimization.add_argument("--warm", action="store_true")
    optimization.add_argument("--warm_epochs", type=int, default=10)
    optimization.add_argument("--ma_update", action="store_true")
    optimization.add_argument("--ma_beta", type=float, default=0.5)

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", type=str, default="icbhi", choices=["icbhi"])
    data.add_argument("--data_folder", type=str, default="./data")
    data.add_argument("--batch_size", type=int, default=32)
    data.add_argument("--num_workers", type=int, default=8)
    data.add_argument("--class_split", type=str, default="lungsound")
    data.add_argument("--n_cls", type=int, choices=[2, 4], required=True)
    data.add_argument(
        "--test_fold",
        type=str,
        default="official",
        choices=["official", "0", "1", "2", "3", "4"],
    )
    data.add_argument("--sample_rate", type=int, default=16000)
    data.add_argument("--desired_length", type=int, default=5)
    data.add_argument("--n_mels", type=int, default=128)
    data.add_argument("--pad_types", type=str, default="repeat")
    data.add_argument("--resz", type=float, default=1.0)
    data.add_argument("--raw_augment", type=int, default=0)
    data.add_argument("--specaug_policy", type=str, default="icbhi_ast_sup")
    data.add_argument("--specaug_mask", type=str, default="mean")
    data.add_argument("--nospec", action="store_true")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model", type=str, default="beats", choices=["beats"])
    model_group.add_argument("--from_sl_official", action="store_true")
    model_group.add_argument("--audioset_pretrained", action="store_true")

    pafa = parser.add_argument_group("pafa")
    pafa.add_argument("--norm_type", type=str, default="ln", choices=["bn", "ln"])
    pafa.add_argument("--hidden_dim", type=int, default=None)
    pafa.add_argument("--output_dim", type=int, default=768)
    pafa.add_argument(
        "--proj_type",
        type=str,
        default="end2end",
        choices=["end2end", "feat_fixed", "proj_fixed"],
    )
    pafa.add_argument("--lambda_pcsl", type=float, default=50.0)
    pafa.add_argument("--lambda_gpal", type=float, default=5e-4)
    pafa.add_argument("--w_pafa", type=float, default=1.0)

    spa = parser.add_argument_group("spa")
    spa.add_argument("--spa_momentum", type=float, default=0.99)
    spa.add_argument("--spa_temperature", type=float, default=2.0)
    spa.add_argument("--spa_lambda", type=float, default=0.5)
    spa.add_argument("--spa_gamma", type=float, default=1.0)
    spa.add_argument("--spa_fusion_weight", type=float, default=0.5)
    spa.add_argument("--spa_scale_init", type=float, default=16.0)
    spa.add_argument("--spa_label_smoothing", type=float, default=0.1)
    spa.add_argument(
        "--spa_kd_direction",
        type=str,
        default="geo_to_sphere",
        choices=["geo_to_sphere", "symmetric"],
    )
    spa.add_argument(
        "--spa_update_after_all_classes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    spa.add_argument("--w_spa", type=float, default=1.0)
    spa.add_argument("--ece_bins", type=int, default=15)

    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.lr_decay_epochs = [
        int(item) for item in str(args.lr_decay_epochs).split(",") if item
    ]
    args.method = "pafa_spa"
    args.d_cls = 2
    args.cls_list = (
        ["normal", "crackle", "wheeze", "both"]
        if args.n_cls == 4
        else ["normal", "abnormal"]
    )
    args.device_list = ["L", "A", "M", "3"]
    args.h = int(args.desired_length * 100 - 2)
    args.w = args.n_mels

    split_name = "official" if args.test_fold == "official" else f"fold{args.test_fold}"
    task_name = f"{args.n_cls}class"
    args.model_name = f"icbhi_beats_pafa_spa_{task_name}_{split_name}_seed{args.seed}"
    if args.tag:
        args.model_name += f"_{args.tag}"
    args.save_folder = os.path.join(args.save_dir, args.model_name)

    if args.warm:
        args.warmup_from = args.learning_rate * 0.1
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
        if args.cosine:
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * args.warm_epochs / args.epochs)
            ) / 2
        else:
            args.warmup_to = args.learning_rate
    return args


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = deterministic
    cudnn.benchmark = True


def clear_legacy_dataset_cache() -> None:
    for path in (Path("./data/training.pt"), Path("./data/test.pt")):
        if path.exists():
            path.unlink()
            print(f"Removed stale PAFA dataset cache: {path}")


def build_loaders(args: argparse.Namespace, *, train_required: bool = True):
    if not args.keep_legacy_cache:
        clear_legacy_dataset_cache()

    train_dataset = None
    if train_required:
        train_dataset = ICBHIDataset(
            train_flag=True, transform=None, args=args, print_flag=True
        )
    val_dataset = ICBHIDataset(
        train_flag=False, transform=None, args=args, print_flag=True
    )
    train_loader = None
    if train_dataset is not None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def build_model(args: argparse.Namespace, device: torch.device):
    kwargs: dict[str, Any] = {}
    kwargs["spec_transform"] = None if args.nospec else SpecAugment(args)
    backbone = get_backbone_class("beats")(**kwargs).to(device)

    projector = ProjectionHead(
        backbone.final_feat_dim,
        args.hidden_dim,
        args.output_dim,
        attention=True,
        norm_type=args.norm_type,
        proj_type=args.proj_type,
    ).to(device)

    spa_head = SPAHead(
        backbone.final_feat_dim,
        args.n_cls,
        momentum=args.spa_momentum,
        temperature=args.spa_temperature,
        fusion_weight=args.spa_fusion_weight,
        scale_init=args.spa_scale_init,
        update_after_all_classes=args.spa_update_after_all_classes,
    ).to(device)

    spa_criterion = SPALoss(
        gamma=args.spa_gamma,
        alignment_weight=args.spa_lambda,
        distillation_temperature=args.spa_temperature,
        label_smoothing=args.spa_label_smoothing,
        kd_direction=args.spa_kd_direction,
    ).to(device)
    pafa_criterion = PAFALoss().to(device)

    parameters = (
        list(backbone.parameters())
        + list(projector.parameters())
        + list(spa_head.parameters())
    )
    optimizer = set_optimizer(args, parameters)
    return backbone, projector, spa_head, spa_criterion, pafa_criterion, optimizer


def snapshot_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


@torch.no_grad()
def apply_moving_average(
    module: nn.Module,
    previous_state: dict[str, torch.Tensor],
    beta: float,
    *,
    parameters_only: bool = False,
) -> None:
    parameter_names = {name for name, _ in module.named_parameters()}
    current = module.state_dict()
    for name, value in current.items():
        if parameters_only and name not in parameter_names:
            continue
        previous = previous_state[name].to(device=value.device, dtype=value.dtype)
        if torch.is_floating_point(value):
            value.mul_(1.0 - beta).add_(previous, alpha=beta)
    module.load_state_dict(current, strict=True)


def train_one_epoch(
    train_loader,
    backbone: nn.Module,
    projector: nn.Module,
    spa_head: SPAHead,
    spa_criterion: SPALoss,
    pafa_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    backbone.train()
    projector.train()
    spa_head.train()

    total_meter = AverageMeter()
    spa_meter = AverageMeter()
    pafa_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    batch_time = AverageMeter()
    end = time.time()
    amp_enabled = args.amp and device.type == "cuda"

    for batch_index, (waveforms, labels) in enumerate(train_loader):
        waveforms = waveforms.to(device, non_blocking=True)
        class_labels = labels[0].to(device, non_blocking=True).long()
        patient_labels = labels[2].to(device, non_blocking=True).long()
        batch_size = class_labels.size(0)

        previous_backbone = previous_projector = previous_spa = None
        if args.ma_update:
            previous_backbone = snapshot_state(backbone)
            previous_projector = snapshot_state(projector)
            previous_spa = snapshot_state(spa_head)

        warmup_learning_rate(
            args, epoch, batch_index, len(train_loader), optimizer
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            temporal_features = backbone(waveforms, training=True)
            spa_output = spa_head(temporal_features)
            projected_features = projector(temporal_features)

            spa_terms = spa_criterion(
                spa_output.logits_s,
                spa_output.logits_g,
                class_labels,
            )
            pafa_loss = pafa_criterion(
                projected_features,
                patient_labels,
                lambda_pcsl=args.lambda_pcsl,
                lambda_gpal=args.lambda_gpal,
            )
            total_loss = args.w_spa * spa_terms.total + args.w_pafa * pafa_loss

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.ma_update:
            assert previous_backbone is not None
            assert previous_projector is not None
            assert previous_spa is not None
            apply_moving_average(backbone, previous_backbone, args.ma_beta)
            apply_moving_average(projector, previous_projector, args.ma_beta)
            apply_moving_average(
                spa_head, previous_spa, args.ma_beta, parameters_only=True
            )

        spa_head.update_geometry(
            spa_output.normalized_features.detach(), class_labels
        )

        predictions = spa_output.fused_probabilities.argmax(dim=1)
        accuracy = (predictions == class_labels).float().mean().item() * 100.0
        total_meter.update(total_loss.item(), batch_size)
        spa_meter.update(spa_terms.total.item(), batch_size)
        pafa_meter.update(pafa_loss.item(), batch_size)
        accuracy_meter.update(accuracy, batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_index + 1) % args.print_freq == 0 or batch_index == 0:
            print(
                f"Train [{epoch:03d}][{batch_index + 1:04d}/{len(train_loader):04d}] "
                f"loss={total_meter.average:.4f} spa={spa_meter.average:.4f} "
                f"pafa={pafa_meter.average:.4f} acc={accuracy_meter.average:.2f} "
                f"time={batch_time.average:.3f}s"
            )
            sys.stdout.flush()

    return {
        "loss": total_meter.average,
        "spa_loss": spa_meter.average,
        "pafa_loss": pafa_meter.average,
        "accuracy": accuracy_meter.average,
    }


@torch.no_grad()
def evaluate(
    val_loader,
    backbone: nn.Module,
    spa_head: SPAHead,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    backbone.eval()
    spa_head.eval()
    amp_enabled = args.amp and device.type == "cuda"

    all_probabilities: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    batch_time = AverageMeter()
    end = time.time()

    for batch_index, (waveforms, labels) in enumerate(val_loader):
        waveforms = waveforms.to(device, non_blocking=True)
        class_labels = labels[0].to(device, non_blocking=True).long()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            temporal_features = backbone(waveforms, training=False)
            spa_output = spa_head(temporal_features)

        all_probabilities.append(
            spa_output.fused_probabilities.float().cpu().numpy()
        )
        all_labels.append(class_labels.cpu().numpy())
        batch_time.update(time.time() - end)
        end = time.time()

        if (batch_index + 1) % args.print_freq == 0:
            print(
                f"Eval [{batch_index + 1:04d}/{len(val_loader):04d}] "
                f"time={batch_time.average:.3f}s"
            )

    probabilities = np.concatenate(all_probabilities, axis=0)
    target = np.concatenate(all_labels, axis=0)
    metrics = compute_classification_metrics(
        probabilities,
        target,
        args.n_cls,
        n_bins=args.ece_bins,
    )
    print(
        "Evaluation "
        f"Sp={metrics['specificity']:.2f} "
        f"Se={metrics['sensitivity']:.2f} "
        f"Score={metrics['score']:.2f} "
        f"Acc={metrics['accuracy']:.2f} "
        f"Macro-F1={metrics['macro_f1']:.2f} "
        f"ECE={metrics['ece']:.2f}"
    )
    return metrics


def is_better(metrics: dict[str, Any], best: dict[str, Any] | None) -> bool:
    # Match the upstream PAFA checkpoint rule: maximize ICBHI Score and
    # reject degenerate checkpoints with near-zero abnormal sensitivity.
    if metrics["sensitivity"] <= 0.1:
        return False
    if best is None:
        return True
    return metrics["score"] > best["score"]


def main() -> None:
    args = finalize_args(build_parser().parse_args())
    seed_everything(args.seed, args.deterministic)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for the PAFA BEATs training configuration. "
            "Install a CUDA-enabled PyTorch build and verify torch.cuda.is_available()."
        )
    device = torch.device("cuda")
    output_dir = Path(args.save_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    args_filename = "eval_args.json" if args.eval else "train_args.json"
    write_json(output_dir / args_filename, vars(args))

    train_loader, val_loader = build_loaders(args, train_required=not args.eval)
    (
        backbone,
        projector,
        spa_head,
        spa_criterion,
        pafa_criterion,
        optimizer,
    ) = build_model(args, device)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    start_epoch = 1
    best_metrics: dict[str, Any] | None = None

    if args.init_pafa_checkpoint:
        initialize_from_pafa_checkpoint(
            Path(args.init_pafa_checkpoint),
            backbone=backbone,
            projector=projector,
        )
        print(f"Initialized backbone/projector from {args.init_pafa_checkpoint}")

    if args.resume:
        checkpoint = load_checkpoint(
            Path(args.resume),
            backbone=backbone,
            projector=projector,
            spa_head=spa_head,
            optimizer=optimizer,
            scaler=scaler,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metrics = checkpoint.get("best_metrics")
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.eval:
        checkpoint_path = args.checkpoint or args.resume
        if checkpoint_path is None:
            raise ValueError("--eval requires --checkpoint or --resume")
        checkpoint = load_checkpoint(
            Path(checkpoint_path),
            backbone=backbone,
            projector=projector,
            spa_head=spa_head,
            strict=True,
        )
        metrics = evaluate(val_loader, backbone, spa_head, args, device)
        metrics["checkpoint"] = str(checkpoint_path)
        metrics["epoch"] = int(checkpoint.get("epoch", -1))
        metrics["method"] = "pafa_spa"
        metrics["n_cls"] = args.n_cls
        metrics["test_fold"] = args.test_fold
        metrics["seed"] = args.seed
        write_json(output_dir / "metrics_eval.json", metrics)
        return

    if train_loader is None:
        raise RuntimeError("training loader was not created")

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        train_metrics = train_one_epoch(
            train_loader,
            backbone,
            projector,
            spa_head,
            spa_criterion,
            pafa_criterion,
            optimizer,
            scaler,
            epoch,
            args,
            device,
        )
        eval_metrics = evaluate(val_loader, backbone, spa_head, args, device)
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "eval": eval_metrics,
        }
        append_jsonl(history_path, row)

        if is_better(eval_metrics, best_metrics):
            best_metrics = deepcopy(eval_metrics)
            best_metrics["epoch"] = epoch
            save_checkpoint(
                output_dir / "best.pth",
                epoch=epoch,
                backbone=backbone,
                projector=projector,
                spa_head=spa_head,
                optimizer=optimizer,
                scaler=scaler,
                best_metrics=best_metrics,
                args=vars(args),
            )
            write_json(output_dir / "metrics_best.json", best_metrics)
            print(
                f"Saved best checkpoint at epoch {epoch}: "
                f"Score={best_metrics['score']:.2f}, ECE={best_metrics['ece']:.2f}"
            )

        if epoch % args.save_freq == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch}.pth",
                epoch=epoch,
                backbone=backbone,
                projector=projector,
                spa_head=spa_head,
                optimizer=optimizer,
                scaler=scaler,
                best_metrics=best_metrics or {},
                args=vars(args),
            )

    if best_metrics is None:
        raise RuntimeError("training ended without a valid checkpoint")
    print(json.dumps(best_metrics, indent=2))


if __name__ == "__main__":
    main()
