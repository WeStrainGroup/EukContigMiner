#!/usr/bin/env python3
"""H1 general 900-config diverse screen: 16 epochs, overall+short Validation."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)


def best_for(sc, y):
    o = np.argsort(-sc); s = sc[o]; yy = y[o].astype(np.int64)
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); p = int(yy.sum()); n = len(yy) - p
    tpr = tp / p if p else np.zeros_like(tp, dtype=float)
    fpr = fp / n if n else np.zeros_like(fp, dtype=float)
    den = .01 * tpr + .99 * fpr
    prec = np.divide(.01 * tpr, den, out=np.ones_like(tpr), where=den > 0)
    f1 = np.divide(2 * prec * tpr, prec + tpr, out=np.zeros_like(tpr), where=(prec + tpr) > 0)
    i = int(np.argmax(f1)); return float(f1[i]), float(tpr[i]), float(fpr[i]), float(s[i])


class H1Flex(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.family = cfg["family"]
        self.inp = nn.BatchNorm1d(943) if cfg["input_norm"] else nn.Identity()
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "mish": nn.Mish}[cfg["activation"]]
        h = cfg["hidden"]; d = cfg["depth"]; dr = cfg["dropout"]
        if self.family == "linear":
            self.net = nn.Linear(943, 1)
        elif self.family == "mlp":
            layers = []
            last = 943
            for _ in range(d):
                layers += [nn.Linear(last, h), act(), nn.Dropout(dr)]
                last = h
            layers += [nn.Linear(last, 1)]
            self.net = nn.Sequential(*layers)
        elif self.family == "residual":
            self.proj = nn.Linear(943, h)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(h, h), act(), nn.Dropout(dr), nn.Linear(h, h), nn.Dropout(dr)) for _ in range(d)])
            self.norm = nn.LayerNorm(h)
            self.out = nn.Linear(h, 1)
        elif self.family == "gated":
            self.proj = nn.Linear(943, 2 * h)
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(h, 2 * h), nn.GLU(), nn.Dropout(dr)) for _ in range(d)])
            self.norm = nn.LayerNorm(h)
            self.out = nn.Linear(h, 1)
        elif self.family == "cnn":
            self.conv = nn.Sequential(nn.Conv1d(1, h, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(h), act(), nn.AdaptiveMaxPool1d(1))
            self.out = nn.Linear(h, 1)
        else:
            raise ValueError(self.family)
        if cfg["init"] == "orthogonal":
            for p in self.parameters():
                if p.ndim >= 2:
                    nn.init.orthogonal_(p)

    def forward(self, x):
        x = self.inp(x)
        if self.family == "linear" or self.family == "mlp":
            return self.net(x).squeeze(-1)
        if self.family == "residual":
            h = self.proj(x)
            for b in self.blocks:
                h = h + b(h)
            return self.out(self.norm(h)).squeeze(-1)
        if self.family == "gated":
            h = self.proj(x)
            h = h[:, :h.shape[1] // 2] * torch.sigmoid(h[:, h.shape[1] // 2:])
            for b in self.blocks:
                h = h + b(h)
            return self.out(self.norm(h)).squeeze(-1)
        if self.family == "cnn":
            h = self.conv(x.unsqueeze(1)).squeeze(-1)
            return self.out(h).squeeze(-1)


def generate_configs(n=900, seed=20261020):
    rng = random.Random(seed)
    out, seen = [], set()
    families = ["mlp"] * 5 + ["residual"] * 3 + ["gated"] * 1 + ["cnn"] * 1 + ["linear"] * 1
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
            "init": rng.choice(["v07", "scratch", "orthogonal"]),
            "label_smoothing": rng.choice([0.0, 0.01, 0.03]),
            "epochs": 16,
        }
        if cfg["family"] == "linear":
            cfg["depth"] = 1
        key = tuple(cfg.values())
        if key not in seen:
            seen.add(key); out.append(cfg)
    return out


@torch.no_grad()
def predict(model, mat, device, batch=16384):
    model.eval(); out = np.empty(len(mat), np.float64)
    for lo in range(0, len(mat), batch):
        hi = min(lo + batch, len(mat))
        x = torch.from_numpy(np.asarray(mat[lo:hi])).to(device, non_blocking=True).float()
        out[lo:hi] = torch.sigmoid(model(x).double()).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=8)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--config-start", type=int, default=0)
    ap.add_argument("--config-end", type=int, default=900)
    args = ap.parse_args()
    F = Path(args.data_root)
    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    train = np.load(F / "train_combined_h1_943.npy", mmap_mode="r")
    val = np.load(F / "validation_h1_943_full.npy", mmap_mode="r")
    ytr = np.load(F / "train_combined_euk_labels.npy").astype(np.float32)
    yva = np.load(F / "validation_euk_labels_full.npy").astype(np.int64)
    loglen = np.load(F / "train_combined_loglen.npy").astype(np.float32)
    vlen = np.load(F / "validation_lengths_full.npy").astype(np.int64)
    cfg_all = [c for i, c in enumerate(generate_configs()) if args.config_start <= i < args.config_end and i % args.num_shards == args.shard]
    done = set()
    if args.skip_existing:
        import csv, glob
        for fp in glob.glob(str(outdir / "results_shard*.csv")):
            for row in csv.DictReader(open(fp)):
                done.add((row["family"], int(row["hidden"]), int(row["depth"]), float(row["dropout"]),
                          row["activation"], row["input_norm"] == "True", float(row["lr"]),
                          float(row["weight_decay"]), int(row["batch_size"]), row["optimizer"],
                          row["schedule"], float(row["short_weight"]), row["init"],
                          float(row["label_smoothing"])))
    log = (outdir / f"shard{args.shard}.log").open("w")
    for cfg in cfg_all:
        key = (cfg["family"], cfg["hidden"], cfg["depth"], cfg["dropout"], cfg["activation"],
               cfg["input_norm"], cfg["lr"], cfg["weight_decay"], cfg["batch_size"], cfg["optimizer"],
               cfg["schedule"], cfg["short_weight"], cfg["init"], cfg["label_smoothing"])
        if key in done:
            continue
        import time
        t0 = time.time()
        cfg["epochs"] = args.epochs
        if cfg["family"] == "cnn":
            cfg["batch_size"] = min(cfg["batch_size"], 4096)
            cfg["hidden"] = min(cfg["hidden"], 512)
        elif cfg["hidden"] >= 1024:
            cfg["batch_size"] = min(cfg["batch_size"], 8192)
        elif cfg["hidden"] >= 512:
            cfg["batch_size"] = min(cfg["batch_size"], 16384)
        else:
            cfg["batch_size"] = min(cfg["batch_size"], 32768)
        try:
            model = H1Flex(cfg).to(args.device)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); print("OOM_INIT", key, flush=True); continue
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
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=args.epochs * math.ceil(len(train) / cfg["batch_size"]))
        w_all = 1.0 + (cfg["short_weight"] - 1.0) * (loglen < np.log10(2000.0)).astype(np.float32)
        w_all = w_all / w_all.mean()
        model.eval()
        for epoch in range(cfg["epochs"]):
            order = np.random.permutation(len(train))
            for lo in range(0, len(train), cfg["batch_size"]):
                idx = order[lo:lo + cfg["batch_size"]]
                x = torch.from_numpy(np.asarray(train[idx], dtype=np.float32)).to(args.device)
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
        pb = 512 if cfg["family"] == "cnn" else 8192
        sc = predict(model, val, args.device, batch=pb)
        f1a, tpra, fpra, thra = best_for(sc, yva)
        sm = vlen < 2000
        f1s, tprs, fprs, thrs = best_for(sc[sm], yva[sm])
        rec = dict(cfg, shard=args.shard, f1_overall=f1a, f1_short=f1s, sec=round(time.time() - t0, 1))
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        with (outdir / f"results_shard{args.shard}.csv").open("a", newline="") as f:
            if f.tell() == 0:
                f.write("family,hidden,depth,dropout,activation,input_norm,lr,weight_decay,batch_size,optimizer,schedule,short_weight,init,label_smoothing,epochs,shard,f1_overall,f1_short,sec\n")
            keys = ("family", "hidden", "depth", "dropout", "activation", "input_norm", "lr", "weight_decay",
                    "batch_size", "optimizer", "schedule", "short_weight", "init", "label_smoothing", "epochs",
                    "shard", "f1_overall", "f1_short", "sec")
            f.write(",".join(str(rec[k]) for k in keys) + "\n")


if __name__ == "__main__":
    main()
