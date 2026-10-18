import csv
import gzip
import io
import json
from pathlib import Path
import numpy as np
import pytest
from eukcontigminer import __version__
from eukcontigminer.cli import main, read_fasta, write_predictions, task_scores, metadata
from eukcontigminer.setup_models import copy_checked


def test_fasta_gzip_and_original_order(tmp_path):
    p = tmp_path/'input.fasta.gz'
    with gzip.open(p, 'wt') as f:
        f.write('>same header\n'+'acgtn'*200+'\n>same header\n'+'RYSWKMBDHV'*100+'\n')
    rows = list(read_fasta(p))
    assert rows == [('same header', b'ACGTN'*200), ('same header', b'RYSWKMBDHV'*100)]


@pytest.mark.parametrize('body', ['', '>x\n', '>x\nACGT\n', '>x\n'+'X'*1000, 'A'*1000])
def test_invalid_fasta(tmp_path, body):
    p = tmp_path/'bad.fa'
    p.write_text(body)
    with pytest.raises(ValueError):
        list(read_fasta(p))


def test_task_decision_boundary_and_batches():
    meta = metadata()
    threshold = meta['tasks']['fun']['threshold']
    class Pipeline:
        sizes = []
        def infer_chunk(self, seq):
            self.sizes.append(len(seq))
            scores = np.full(len(seq), threshold)
            scores[-1] = np.nextafter(threshold, 1.)
            return dict(task='p_fungi', lengths=[len(x) for x in seq], scores=scores)
    pipeline = Pipeline()
    stream = io.StringIO()
    assert write_predictions(((str(i), b'A'*1000) for i in range(2050)), pipeline, stream, 'fun', meta) == (2050, 2)
    rows = list(csv.DictReader(io.StringIO(stream.getvalue()), delimiter='\t'))
    assert pipeline.sizes == [2048, 2]
    assert rows[0]['label'] == 'Other' and rows[2047]['label'] == rows[-1]['label'] == 'Fungi'
    assert rows[2048]['label'] == 'Other' and rows[-1]['contig_id'] == '2049'
    with pytest.raises(ValueError):
        task_scores(dict(task='p_euk', scores=np.ones(1), lengths=[1000]), [1000], 'fun', meta)


def test_no_overwrite_or_silent_bad_task(tmp_path):
    out = tmp_path/'exists.tsv'
    out.write_text('keep')
    with pytest.raises(SystemExit):
        main(['absent.fa', '--task', 'fun', '-o', str(out)])
    assert out.read_text() == 'keep'
    with pytest.raises(SystemExit):
        main(['absent.fa', '--task', 'fungi', '-o', str(out)])


def test_model_setup_rejects_corruption(tmp_path):
    p = tmp_path/'bad.pt'
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        copy_checked(p, tmp_path/'destination.pt', '0'*64)
    assert not (tmp_path/'destination.pt').exists()


def test_release_contract():
    m = metadata()
    assert __version__ == m['version'] == '0.7'
    assert set(m['tasks']) == {'euk', 'fun'}
    assert m['tasks']['euk']['threshold'] == .5
    assert m['tasks']['fun']['threshold'] == 0.9920285346180445
    assert m['models_frozen'] and m['chunk_rows'] == 2048
