"""CLI integration tests use deterministic scores, without downloading model weights."""
import csv
import json
import sys
import types

import numpy as np
import pytest

from eukcontigminer import cli
from eukcontigminer.resources import sha


@pytest.fixture
def fake_runtime(monkeypatch):
    state = {'devices': [], 'chunks': [], 'calls': []}
    module = types.ModuleType('eukcontigminer.runtime.encoders')
    module.check_runtime_dependencies = lambda: None
    module.EncoderBundle = lambda root, device, verified: state['devices'].append(device)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli, 'verify_models', lambda root: (root, cli.metadata()))

    class Pipeline:
        def __init__(self, task):
            self.task = task

        def infer_chunk(self, seqs):
            state['calls'].append((self.task, [len(seq) for seq in seqs]))
            scores = {'euk': {65: .9, 67: .5, 71: .8, 84: .1},
                      'fun': {65: .2, 67: .9, 71: .8, 84: .1}}
            return dict(scores=np.array([scores[self.task][seq[0]] for seq in seqs]),
                        lengths=[len(seq) for seq in seqs], routes=np.zeros(len(seqs), dtype=int))

    monkeypatch.setattr(cli, 'load_pipeline', lambda root, meta, task, device, encoders, threads: Pipeline(task))
    monkeypatch.setattr(cli, 'infer_pipelines',
                        lambda pipelines, seqs: {task: pipe.infer_chunk(seqs) for task, pipe in pipelines.items()})
    return state


def make_input(tmp_path):
    path = tmp_path / 'input.fa'
    path.write_text('>short\n' + 'a' * 499 + '\n>at500\n' + 'a' * 500 +
                    '\n>at999\n' + 'a' * 999 + '\n>same header\n' + 'a' * 1000 +
                    '\n>same header\n' + 'c' * 1000 + '\n>both\n' + 'g' * 1001 +
                    '\n>negative\n' + 't' * 1000 + '\n')
    return path


@pytest.mark.parametrize('tasks,files', [('euk', {'euk.fa'}), ('fun', {'fun.fa'}),
                                      ('all', {'euk.fa', 'fun.fa'}), ('both', {'euk.fa', 'fun.fa'})])
def test_directory_outputs_match_labels_and_duplicate_records(tmp_path, fake_runtime, tasks, files):
    source = make_input(tmp_path)
    output = tmp_path / 'nested' / 'results'
    cli.main(['--input', str(source), '--output', str(output), '--task', tasks, '--device', 'cpu'])
    assert {p.name for p in output.iterdir()} == files | {'ecm.tsv', 'run.json'}
    with (output / 'ecm.tsv').open() as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    original = list(cli.read_fasta(source))
    assert [row['contig_id'] for row in rows] == [name for name, seq in original]
    for filename in files:
        task = filename[:-3]
        positive = 'eukaryota' if task == 'euk' else 'fungi'
        expected = [record for row, record in zip(rows, original) if row[f'label_{task}'] == positive]
        assert list(cli.read_fasta(output / filename)) == expected
        assert all(row[f'label_{task}'] == 'NA' for row in rows[:3])
    summary = json.loads((output / 'run.json').read_text())
    assert summary['version'] == '0.91' and summary['status'] == 'completed'
    assert summary['shared_features'] == (len(files) == 2)
    assert summary['min_input_bp'] == 1000 and summary['chunk_size_requested'] == 'auto'
    assert summary['chunk_rows'] == 2048 and summary['device'] == 'cpu'
    assert fake_runtime['devices'] == ['cpu']
    assert summary['rows'] == 7 and summary['input_sha256'] == sha(source)
    for filename, details in summary['outputs'].items():
        assert details['sha256'] == sha(output / filename)
        assert details['bytes'] == (output / filename).stat().st_size
    assert all(count == 2 for count in summary['selected_contigs'].values())
    if tasks == 'all':
        assert rows[4]['label_euk'] == 'other' and rows[4]['label_fun'] == 'fungi'


def test_explicit_500_gpu_and_chunk_options(tmp_path, fake_runtime, monkeypatch):
    source = make_input(tmp_path)
    output = tmp_path / 'result'
    # Check the public gpu option resolves before hardware-based chunk sizing.
    def resolve(requested, device):
        assert requested == '3' and device == 'cuda:0'
        return 3
    monkeypatch.setattr(cli, 'resolve_chunk_size', resolve)
    cli.main(['--input', str(source), '--output', str(output), '--min-length', '500', '--chunk-size', '3'])
    result = json.loads((output / 'run.json').read_text())
    assert result['selected_contigs'] == {'euk': 4}
    assert result['chunks'] == 3 and result['min_input_bp'] == 500
    assert result['device_requested'] == 'gpu' and fake_runtime['devices'] == ['cuda:0']


def test_empty_positive_fasta_is_a_valid_completed_output(tmp_path, fake_runtime):
    source = tmp_path / 'negative.fa'
    source.write_text('>negative\n' + 'T' * 1000 + '\n>short\nA\n')
    output = tmp_path / 'results'
    cli.main(['--input', str(source), '--output', str(output), '--task', 'all', '--device', 'cpu'])
    assert (output / 'euk.fa').read_bytes() == (output / 'fun.fa').read_bytes() == b''
    assert json.loads((output / 'run.json').read_text())['selected_contigs'] == {'euk': 0, 'fun': 0}


def test_failed_late_record_leaves_no_result_directory(tmp_path, fake_runtime):
    source = tmp_path / 'bad.fa'
    source.write_text('>valid\n' + 'A' * 1000 + '\n>bad\nXYZ\n')
    output = tmp_path / 'results'
    with pytest.raises(ValueError, match='IUPAC'):
        cli.main(['--input', str(source), '--output', str(output), '--device', 'cpu', '--chunk-size', '1'])
    assert not output.exists()


def test_existing_directory_is_never_overwritten(tmp_path, fake_runtime):
    output = tmp_path / 'results'
    output.mkdir()
    (output / 'euk.fa').write_text('keep')
    with pytest.raises(SystemExit):
        cli.main(['--input', 'absent.fa', '--output', str(output), '--device', 'cpu'])
    assert (output / 'euk.fa').read_text() == 'keep'


@pytest.mark.parametrize('options', [['--min-length', '499'], ['--chunk-size', '0'],
                                   ['--device', 'cuda:0'], ['--cpu-threads', '0']])
def test_invalid_public_options_are_rejected(tmp_path, options):
    with pytest.raises(SystemExit):
        cli.main(['--input', 'absent.fa', '--output', str(tmp_path / 'results'), *options])


def test_gzip_input_exports_the_same_records(tmp_path, fake_runtime):
    import gzip
    source = make_input(tmp_path)
    compressed = tmp_path / 'input.fa.gz'
    with gzip.open(compressed, 'wb') as stream:
        stream.write(source.read_bytes())
    output = tmp_path / 'results'
    cli.main(['--input', str(compressed), '--output', str(output), '--task', 'all', '--device', 'cpu'])
    assert list(cli.read_fasta(output / 'fun.fa')) == [('same header', b'C' * 1000), ('both', b'G' * 1001)]


def test_changed_input_aborts_without_publishing(tmp_path, fake_runtime, monkeypatch):
    source = make_input(tmp_path)
    original_reader = cli.read_fasta
    def changing_reader(path):
        yield from original_reader(path)
        with path.open('a') as stream:
            stream.write('N\n')
    monkeypatch.setattr(cli, 'read_fasta', changing_reader)
    output = tmp_path / 'results'
    with pytest.raises(ValueError, match='FASTA changed'):
        cli.main(['--input', str(source), '--output', str(output), '--device', 'cpu'])
    assert not output.exists()


def test_publication_failure_rolls_back_this_run(tmp_path, fake_runtime, monkeypatch):
    source = make_input(tmp_path)
    original_hash = sha(source)
    original_link = cli.os.link
    def fail_final_link(src, dst):
        if dst.name == 'run.json':
            raise OSError('simulated publication failure')
        return original_link(src, dst)
    monkeypatch.setattr(cli.os, 'link', fail_final_link)
    output = tmp_path / 'results'
    with pytest.raises(OSError, match='publication failure'):
        cli.main(['--input', str(source), '--output', str(output), '--task', 'all', '--device', 'cpu'])
    assert not output.exists() and sha(source) == original_hash
