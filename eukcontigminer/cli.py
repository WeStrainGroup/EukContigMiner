from __future__ import annotations

import argparse
import json
from pathlib import Path

from .deployment import predict_fasta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score whole contigs with the frozen DNA + ESM-2 model"
    )
    parser.add_argument("fasta", type=Path, help="input FASTA or FASTA.GZ")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--config", type=Path, help="advanced: override bundled model config"
    )
    parser.add_argument("--device", default="cuda:0")
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
        help="disable DNA early exit and send every eligible contig through ESM",
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
