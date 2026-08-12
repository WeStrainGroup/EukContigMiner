from __future__ import annotations

import argparse
import csv
import gzip
import sys
from contextlib import nullcontext
from pathlib import Path

from Bio import SeqIO

from .predictor import DEFAULT_THRESHOLD, Predictor


def _open_fasta(path: str):
    if path == "-":
        return nullcontext(sys.stdin)
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify FASTA contigs as Eukaryota or Other")
    parser.add_argument("fasta", help="input FASTA/FASTA.GZ, or - for stdin")
    parser.add_argument("-o", "--output", default="-", help="TSV output, default stdout")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default=None, help="cpu, cuda, or a torch device")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.batch_size < 1:
        parser.error("threshold must be in [0,1] and batch-size must be positive")
    with _open_fasta(args.fasta) as handle:
        records = list(SeqIO.parse(handle, "fasta"))
    predictor = Predictor(args.device)
    probabilities = predictor.predict((str(record.seq) for record in records), args.batch_size)
    output = sys.stdout if args.output == "-" else open(args.output, "w", newline="")
    try:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("contig", "length", "p_euk", "label"))
        for record, probability in zip(records, probabilities, strict=True):
            label = "Eukaryota" if probability > args.threshold else "Other"
            writer.writerow((record.id, len(record.seq), f"{probability:.8f}", label))
    finally:
        if output is not sys.stdout:
            output.close()


if __name__ == "__main__":
    main()
