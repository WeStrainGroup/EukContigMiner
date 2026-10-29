#!/usr/bin/env python3
"""Generic H2/H3 1000-config screen: full data, 16 epoch, overall+short."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_h1_900 import best_for, predict  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))


class HeadFlex(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        self.family = cfg["family"]
        self.inp = nn.BatchNorm1d(input_dim) if cfg["input_norm"] else nn.Identity()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "mish": nn.Mish}[cfg["activation"]]
        h, d, dr = cfg["hidden"], cfg["depth"], cfg["dropout"]
        if self.family == "linear":
            self.net = nn.Linear(input_dim, 1)
        elif self.family == "mlp":
            layers, last = [], input_dim
            for _ in range(d):
                layers += [nn.Linear(last, h), act(), nn.Dropout(dr)]
                last = h
            layers += [nn.Linear(last, 1)]
            self.net = nn.Sequential(*layers)
        elif self.family == "residual":
            self.proj = nn.Linear(input_dim, h)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(h, h), act(), nn.Dropout(dr), nn.Linear(h, h), nn.Dropout(dr)) for _ in range(d)])
            self.norm = nn.LayerNorm(h); self.out = nn.Linear(h, 1)
        elif self.family == "gated":
            self.proj = nn.Linear(input_dim, 2 * h)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.GLU(), nn.Dropout(dr)) for _ in range(d)])
            self.norm = nn.LayerNorm(h); self.out = nn.Linear(h, 1)
        else:
            raise ValueError(self.family)
        if cfg["init"] == "orthogonal":
            for p in self.parameters():
                if p.ndim >= 2:
                    nn.init.orthogonal_(p)

    def forward(self, x):
        x = self.inp(x)
        if self.family in ("linear", "mlp"):
            return self.net(x).squeeze(-1)
        if self.family == "residual":
            h = self.proj(x)
            for b in self.blocks:
                h = h + b(h)
            return self.out(self.norm(h)).squeeze(-1)
        h = self.proj(x)
        h = h[:, :h.shape[1] // 2] * torch.sigmoid(h[:, h.shape[1] // 2:])
        for b in self.blocks:
            h = h + b(h)
        return self.out(self.norm(h)).squeeze(-1)


def effective_batch(cfg):
    bs = int(cfg["batch_size"]); hidden = int(cfg["hidden"])
    if hidden >= 1024:
        bs = min(bs, 8192)
    elif hidden >= 512:
        bs = min(bs, 16384)
    else:
        bs = min(bs, 32768)
    return bs


def key_cfg(cfg):
    return (cfg["family"], int(cfg["hidden"]), int(cfg["depth"]), float(cfg["dropout"]), cfg["activation"],
            bool(cfg["input_norm"]), float(cfg["lr"]), float(cfg["weight_decay"]), effective_batch(cfg),
            cfg["optimizer"], cfg["schedule"], float(cfg["short_weight"]), cfg["init"], float(cfg["label_smoothing"]))


def generate_configs(n=1000, seed=20261023):
    rng = random.Random(seed)
    families = ["mlp"] * 5 + ["residual"] * 3 + ["gated"] * 2 + ["linear"] * 1
    out, seen = [], set()
    while len(out) < n:
        cfg = {
            "family": rng.choice(families),
            "hidden": rng.choice([128, 192, 256, 384, 512, 768, 1024, 1536, 2048]),
            "depth": rng.choice([1, 2, 3, 4]),
            "dropout": rng.choice([0.0, 0.05, 0.1, 0.2, 0.3, 0.4]),
            "activation": rng.choice(["gelu", "relu", "silu", "mish"]),
            "input_norm": rng.choice([True, False]),
            "lr": float(10 ** rng.uniform(math.log10(1e-5), math.log10(3e-3))),
            "weight_decay": rng.choice([0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
            "batch_size": rng.choice([8192, 16384, 32768, 65536]),
            "optimizer": rng.choice(["adamw", "adam", "sgd"]),
            "schedule": rng.choice(["cosine", "step", "onecycle"]),
            "short_weight": rng.choice([1.0, 2.0, 4.0, 8.0, 16.0]),
            "init": rng.choice(["scratch", "orthogonal"]),
            "label_smoothing": rng.choice([0.0, 0.01, 0.03]),
        }
        if cfg["family"] == "linear":
            cfg["depth"] = 1
        key = tuple(cfg.values())
        if key not in seen:
            seen.add(key); out.append(cfg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--val-file", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--loglen", required=True)
    ap.add_argument("--vlen", required=True)
    ap.add_argument("--input-dim", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--config-start", type=int, default=0)
    ap.add_argument("--config-end", type=int, default=1000)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--block-size", type=int, default=0)
    ap.add_argument("--in-memory", action="store_true")
    ap.add_argument("--max-batches", type=int, default=0)
    args = ap.parse_args()
    F = Path(args.data_root); outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    mode = None if args.in_memory else "r"
    print("STAGE load_start", flush=True)
    train = np.load(F / args.train_file, mmap_mode=mode)
    val = np.load(F / args.val_file, mmap_mode=mode)
    print("STAGE load_done", train.shape, val.shape, flush=True)
    ytr = np.load(F / args.labels).astype(np.float32)
    yva = np.load(F / "validation_euk_labels_full.npy").astype(np.int64)
    loglen = np.load(F / args.loglen).astype(np.float32)
    vlen = np.load(F / "validation_lengths_full.npy").astype(np.int64)
    cfg_all = [c for i, c in enumerate(generate_configs()) if args.config_start <= i < args.config_end and i % args.num_shards == args.shard]
    done = set()
    fp = outdir / f"results_shard{args.shard}.csv"
    if args.skip_existing and fp.exists():
        for r in csv.DictReader(open(fp)):
            done.add((r["family"], int(r["hidden"]), int(r["depth"]), float(r["dropout"]), r["activation"], r["input_norm"] == "True", float(r["lr"]), float(r["weight_decay"]), int(r["batch_size"]), r["optimizer"], r["schedule"], float(r["short_weight"]), r["init"], float(r["label_smoothing"])))
    log = (outdir / f"shard{args.shard}.log").open("a")
    for cfg in cfg_all:
        key = key_cfg(cfg)
        if key in done:
            continue
        cfg = dict(cfg, input_dim=args.input_dim, epochs=args.epochs)
        if cfg["hidden"] >= 1024:
            cfg["batch_size"] = min(cfg["batch_size"], 8192)
        elif cfg["hidden"] >= 512:
            cfg["batch_size"] = min(cfg["batch_size"], 16384)
        else:
            cfg["batch_size"] = min(cfg["batch_size"], 32768)
        t0 = __import__("time").time()
        print("STAGE model_start", cfg["family"], cfg["hidden"], cfg["depth"], flush=True)
        model = HeadFlex(args.input_dim, cfg).to(args.device)
        print("STAGE model_done", flush=True)
        opt = (torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
               if cfg["optimizer"] == "adamw" else
               torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
               if cfg["optimizer"] == "adam" else
               torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"]))
        if cfg["schedule"] == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        elif cfg["schedule"] == "step":
            sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 3), gamma=0.3)
        else:
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=args.epochs * int(np.ceil(len(train) / cfg["batch_size"])))
        w_all = 1.0 + (cfg["short_weight"] - 1.0) * (loglen < np.log10(2000.0)).astype(np.float32)
        w_all = w_all / w_all.mean()
        model.eval()
        for epoch in range(args.epochs):
            if args.block_size:
                blocks = list(range(0, len(train), args.block_size))
                np.random.shuffle(blocks)
                order = np.concatenate([np.arange(b, min(b + args.block_size, len(train)), dtype=np.int64) for b in blocks])
            else:
                order = np.random.permutation(len(train))
            for bi, lo in enumerate(range(0, len(train), cfg["batch_size"])):
                if args.max_batches and bi >= args.max_batches:
                    break
                if bi == 0:
                    print("STAGE first_batch", epoch + 1, flush=True)
                idx = order[lo:lo + cfg["batch_size"]]
                x = torch.from_numpy(np.asarray(train[idx, :args.input_dim])).to(args.device, non_blocking=True).float()
                y = torch.from_numpy(ytr[idx]).to(args.device)
                w = torch.from_numpy(w_all[idx]).to(args.device)
                if cfg["label_smoothing"]:
                    y = y * (1 - cfg["label_smoothing"]) + 0.5 * cfg["label_smoothing"]
                loss = (nn.functional.binary_cross_entropy_with_logits(model(x), y, reduction="none") * w).mean()
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
                if cfg["schedule"] == "onecycle":
                    sched.step()
            if cfg["schedule"] != "onecycle":
                sched.step()
            print("STAGE epoch_done", epoch + 1, flush=True)
            print(json.dumps({"epoch": epoch + 1, "tag": cfg.get("family"), "lr": cfg["lr"]}), flush=True)
        sc = predict(model, val[:, :args.input_dim], args.device, batch=8192)
        f1a, *_ = best_for(sc, yva)
        sm = vlen < 2000
        f1s, *_ = best_for(sc[sm], yva[sm])
        rec = dict(cfg, shard=args.shard, f1_overall=f1a, f1_short=f1s, sec=round(__import__("time").time() - t0, 1))
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        keys = ("family", "hidden", "depth", "dropout", "activation", "input_norm", "lr", "weight_decay", "batch_size", "optimizer", "schedule", "short_weight", "init", "label_smoothing", "input_dim", "epochs", "shard", "f1_overall", "f1_short", "sec")
        with fp.open("a", newline="") as f:
            if f.tell() == 0:
                f.write(",".join(keys) + "\n")
            f.write(",".join(str(rec[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
