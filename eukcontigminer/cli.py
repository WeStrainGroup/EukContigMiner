"""Public offline CLI over the two frozen, independently validated runtimes."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from . import __version__
from .resources import metadata, sha, verify_models

_TORCH_CONFIGURED = False


def read_fasta(path, min_length=1):
    opener = gzip.open if str(path).endswith('.gz') else open
    header, parts = None, []
    allowed = frozenset(b'ACGTRYSWKMBDHVN')
    with opener(path, 'rb') as f:
        for line in f:
            line = line.rstrip(b'\r\n')
            if not line:
                continue
            if line.startswith(b'>'):
                if header is not None:
                    yield checked_record(header, parts, min_length)
                header, parts = line[1:].decode('utf-8'), []
                if not header.strip() or any(ord(c) < 32 or ord(c) == 127 for c in header):
                    raise ValueError('Invalid FASTA header')
            else:
                line = line.upper()
                if header is None or not set(line) <= allowed:
                    raise ValueError('Expected IUPAC DNA FASTA')
                parts.append(line)
    if header is None:
        raise ValueError('Empty FASTA input')
    yield checked_record(header, parts, min_length)


def checked_record(header, parts, min_length=1):
    seq = b''.join(parts)
    if len(seq) < min_length:
        raise ValueError(f'Contig {header!r} is shorter than the supported minimum of {min_length} bp')
    return header, seq


def batches(records):
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == 2048:
            yield batch
            batch = []
    if batch:
        yield batch


def load_pipeline(root, meta, task, device, encoders=None):
    import torch
    global _TORCH_CONFIGURED
    if not _TORCH_CONFIGURED:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _TORCH_CONFIGURED = True
    if task == 'fun':
        from .runtime.fun import FunPipeline
        return FunPipeline(root, meta['files']['fun/head.pt']['sha256'], device, encoders=encoders)
    from .runtime.euk import EukPipeline
    return EukPipeline(root, meta['euk_head_thresholds'], meta['euk_gates'], device, encoders=encoders)


def task_scores(raw, lengths, task, meta):
    import numpy as np
    if not np.array_equal(raw['lengths'], lengths):
        raise ValueError('Sequence lengths changed')
    if task == 'fun':
        if raw.get('task') != 'p_fungi':
            raise ValueError('Incorrect task runtime')
        scores = np.asarray(raw['scores'])
    else:
        if 'scores' not in raw:
            raise ValueError('Euk runtime did not return scores')
        scores = np.asarray(raw['scores'])
    if scores.shape != (len(lengths),) or not np.isfinite(scores).all() or not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError('Invalid prediction scores')
    return scores


def write_predictions(records, pipelines, stream, tasks, meta):
    import numpy as np
    writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
    writer.writerow(['contig_id', 'length_bp', 'p_euk', 'label_euk', 'head_euk', 'p_fun', 'label_fun', 'head_fun'])
    total = count = 0
    for batch in batches(records):
        lengths = [len(seq) for _, seq in batch]
        scores = {}
        heads = {}
        for task in tasks:
            scores[task] = np.full(len(batch), np.nan, dtype=np.float64)
            heads[task] = np.full(len(batch), 'NA', dtype=object)
            min_bp = meta['min_input_bp'][task]
            eligible = [i for i, n in enumerate(lengths) if n >= min_bp]
            if eligible:
                raw = pipelines[task].infer_chunk([batch[i][1] for i in eligible])
                scores[task][eligible] = task_scores(raw, [lengths[i] for i in eligible], task, meta)
                if task == 'euk':
                    routes = raw.get('routes')
                    if routes is None:
                        raise ValueError('Euk runtime did not return routes')
                    routes = np.asarray(routes)
                    if routes.shape != (len(eligible),):
                        raise ValueError('Euk routes must match eligible rows')
                    if not np.issubdtype(routes.dtype, np.integer) or not np.isin(routes, (0, 1, 2)).all():
                        raise ValueError('Euk routes must be integers 0/1/2')
                    for j, route in zip(eligible, routes):
                        heads['euk'][j] = ('1', '2', '3')[int(route)]
                else:
                    heads['fun'][eligible] = '1'
        for i, (header, seq) in enumerate(batch):
            row = [header, len(seq)]
            for task in ('euk', 'fun'):
                if task not in tasks or not np.isfinite(scores[task][i]):
                    row.extend(['NA', 'NA', 'NA'])
                else:
                    score = float(scores[task][i])
                    label = ('eukaryota' if task == 'euk' else 'fungi') if score > meta['tasks'][task]['threshold'] else 'other'
                    row.extend([format(score, '.17g'), label, heads[task][i]])
            writer.writerow(row)
        total += len(batch)
        count += 1
        stream.flush()
    return total, count

def parse_tasks(values):
    if not values:
        return ['euk']
    out = []
    for value in values:
        for token in value.split(','):
            token = token.strip().lower()
            if token in ('both', 'euk,fun', 'fun,euk'):
                for task in ('euk', 'fun'):
                    if task not in out:
                        out.append(task)
            elif token in ('euk', 'fun'):
                if token not in out:
                    out.append(token)
            else:
                raise ValueError('task must be euk, fun, both, or a comma-separated combination')
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'download-models':
        from .setup_models import main as setup
        return setup(argv[1:])
    meta = metadata()
    parser = argparse.ArgumentParser(description=f"Classify euk/fun contigs; minimum {meta['min_input_bp']['euk']} bp (euk) / {meta['min_input_bp']['fun']} bp (fun); shorter records are emitted with NA.")
    parser.add_argument('--version', action='version', version='%(prog)s ' + __version__)
    parser.add_argument('fasta', help='DNA FASTA or .fasta.gz')
    parser.add_argument('--task', action='append', default=None, help='euk (default), fun, both, or euk,fun')
    parser.add_argument('--models', default=os.environ.get('EUKCONTIGMINER_MODELS', 'models'), help='Directory from download-models')
    parser.add_argument('-o', '--output', required=True, help='New TSV file')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 (default) or cpu')
    args = parser.parse_args(argv)
    try:
        tasks = parse_tasks(args.task)
    except ValueError as exc:
        parser.error(str(exc))
    output = Path(args.output).resolve()
    summary = Path(str(output) + '.summary.json')
    partial = Path(str(output) + '.partial')
    for p in [output, summary, partial]:
        if p.exists():
            parser.error('Output already exists: ' + str(p))
    source = Path(args.fasta).resolve()
    # Parse once before loading models; no partial prediction for malformed inputs.
    rows_expected = sum(1 for _ in read_fasta(source))
    from .runtime.encoders import check_runtime_dependencies
    check_runtime_dependencies()
    root, meta = verify_models(args.models)
    before = sha(source)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    start = time.monotonic()
    from .runtime.encoders import EncoderBundle
    encoders = EncoderBundle(root, args.device, verified=True)
    pipelines = {task: load_pipeline(root, meta, task, args.device, encoders) for task in tasks}
    with partial.open('x', newline='') as stream:
        rows, chunks = write_predictions(read_fasta(source), pipelines, stream, tasks, meta)
    if rows != rows_expected or sha(source) != before:
        raise ValueError('FASTA changed during inference')
    # Hard-link publishes without overwriting a file created concurrently.
    os.link(partial, output)
    partial.unlink()
    result = dict(version=__version__, tasks=tasks, rows=rows, chunks=chunks,
                  thresholds={t: meta['tasks'][t]['threshold'] for t in tasks}, strict_greater_than=True,
                  model_ids={t: meta['tasks'][t]['model_id'] for t in tasks}, device=args.device,
                  input_sha256=before, output_sha256=sha(output), seconds=time.monotonic()-start)
    with summary.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
