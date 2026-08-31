from __future__ import annotations

import argparse
import json
from pathlib import Path

from .deployment import predict_fasta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score whole contigs with the frozen DNA + ESM-C 300M model"
    )
    parser.add_argument("fasta", type=Path, help="input FASTA or FASTA.GZ")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--config", type=Path, help="advanced: override bundled model config"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="inference device: auto, cpu, cuda, or cuda:N (default: auto)",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="maximum PyTorch CPU threads (for a 32-core server, use 32)",
    )
    parser.add_argument("--buffer-records", type=int, default=4096)
    parser.add_argument("--dna-batch-size", type=int, default=32)
    parser.add_argument("--dna-max-padded-bases", type=int, default=800000)
    parser.add_argument(
        "--min-length",
        type=int,
        default=1000,
        help="omit contigs shorter than this many base pairs (default: 1000)",
    )
    parser.add_argument(
        "--full-esm",
        action="store_true",
        help="disable DNA early exit and send every eligible contig through ESM-C",
    )
    args = parser.parse_args(argv)
    summary = args.summary or args.output.with_name(
        args.output.name + ".summary.json"
    )
    payload = predict_fasta(
        args.fasta,
        args.output,
        summary,
        config=args.config,
        device_name=args.device,
        cpu_threads=args.cpu_threads,
        buffer_records=args.buffer_records,
        dna_batch_size=args.dna_batch_size,
        dna_max_padded_bases=args.dna_max_padded_bases,
        use_dna_early_exit=not args.full_esm,
        minimum_length=args.min_length,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
