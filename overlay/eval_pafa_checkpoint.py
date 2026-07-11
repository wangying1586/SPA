from __future__ import annotations

import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from models import get_backbone_class
from training.checkpoint import write_json
from training.metrics import compute_classification_metrics
from util.augmentation import SpecAugment
from util.icbhi_dataset import ICBHIDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate an official PAFA checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data_folder", default="./data")
    parser.add_argument("--n_cls", type=int, choices=[2, 4], required=True)
    parser.add_argument(
        "--test_fold",
        default="official",
        choices=["official", "0", "1", "2", "3", "4"],
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--desired_length", type=int, default=5)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--pad_types", default="repeat")
    parser.add_argument("--raw_augment", type=int, default=0)
    parser.add_argument("--resz", type=float, default=1.0)
    parser.add_argument("--specaug_policy", default="icbhi_ast_sup")
    parser.add_argument("--specaug_mask", default="mean")
    parser.add_argument("--nospec", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ece_bins", type=int, default=15)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.dataset = "icbhi"
    args.model = "beats"
    args.class_split = "lungsound"
    args.cls_list = (
        ["normal", "crackle", "wheeze", "both"]
        if args.n_cls == 4
        else ["normal", "abnormal"]
    )
    args.device_list = ["L", "A", "M", "3"]
    args.d_cls = 2
    args.h = int(args.desired_length * 100 - 2)
    args.w = args.n_mels
    return args


def strip_prefix(state_dict):
    result = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("backbone."):
            key = key[9:]
        result[key] = value
    return result


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BEATs evaluation")
    device = torch.device("cuda")

    for cache in (Path("./data/training.pt"), Path("./data/test.pt")):
        if cache.exists():
            cache.unlink()

    dataset = ICBHIDataset(
        train_flag=False, transform=None, args=args, print_flag=True
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    kwargs = {"spec_transform": None if args.nospec else SpecAugment(args)}
    backbone = get_backbone_class("beats")(**kwargs).to(device)
    classifier = nn.Linear(backbone.final_feat_dim, args.n_cls).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    backbone.load_state_dict(strip_prefix(checkpoint["model"]), strict=False)
    classifier.load_state_dict(checkpoint["classifier"], strict=True)
    backbone.eval()
    classifier.eval()

    probabilities = []
    labels_all = []
    amp_enabled = args.amp and device.type == "cuda"
    for waveforms, labels in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        class_labels = labels[0].to(device, non_blocking=True).long()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            features = backbone(waveforms, training=False)
            logits = classifier(features).mean(dim=1)
            probs = torch.softmax(logits, dim=1)
        probabilities.append(probs.float().cpu().numpy())
        labels_all.append(class_labels.cpu().numpy())

    metrics = compute_classification_metrics(
        np.concatenate(probabilities),
        np.concatenate(labels_all),
        args.n_cls,
        n_bins=args.ece_bins,
    )
    metrics.update(
        {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "method": "pafa",
            "n_cls": args.n_cls,
            "test_fold": args.test_fold,
            "seed": args.seed,
        }
    )
    write_json(Path(args.output), metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
