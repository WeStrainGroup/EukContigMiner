"""Joint routing must preserve independent task results and reuse compatible batches."""
from types import SimpleNamespace
import numpy as np
import pytest

from eukcontigminer.runtime import euk


@pytest.fixture
def pipelines(monkeypatch):
    calls = {name: [] for name in ('dna', 'nt', 'esm')}
    def features(name, seqs, width):
        ids = [len(s) - 1000 for s in seqs]
        calls[name].append(ids)
        x = np.zeros((len(seqs), width), np.float32)
        x[:, 0] = ids
        return x
    budgets = []
    def dna(model, seqs, budget):
        budgets.append(budget)
        return features('dna', seqs, 942)
    monkeypatch.setattr(euk, 'encode_dna', dna)
    encoders = SimpleNamespace(dna=lambda: None)
    # Include opposite labels, disjoint/overlapping late stages and exact ties.
    probabilities = {
        'euk': [[.01, .5, .5, .99, .5, .5], [.5, .01, .5, .5, .5, .99], [.5, .5, .8, .5, .5, .5]],
        'fun': [[.5, .99, .5, .5, .5, .5], [.5, .5, .99, .01, .5, .5], [.8, .5, .5, .5, .5, .2]],
    }
    result = {}
    for task, budget in [('euk', 512000), ('fun', 128000)]:
        def head(name, matrix, task=task):
            ids = matrix[:, 0].astype(int)
            if name != 'H1':
                np.testing.assert_array_equal(matrix[:, 942], ids)
            if name == 'H3':
                np.testing.assert_array_equal(matrix[:, 2478], ids)
            return np.asarray(probabilities[task][int(name[-1])-1])[ids]
        pipe = euk.EukPipeline.__new__(euk.EukPipeline)
        pipe.encoders, pipe.token_budget = encoders, budget
        pipe.heads = SimpleNamespace(t=np.array([.5]*3), gates={'H1': [-1., 1.], 'H2': [-1., 1.]}, head_prob=head)
        pipe._nt = lambda seqs: features('nt', seqs, 1536)
        pipe._esm = lambda seqs: features('esm', seqs, 1920)
        result[task] = pipe
    return result, calls, budgets


@pytest.mark.parametrize('order', [('euk', 'fun'), ('fun', 'euk')])
def test_joint_matches_single_and_preserves_batch_context(pipelines, order):
    pipes, calls, budgets = pipelines
    seqs = [b'A'*(1000+i) for i in range(6)]
    separate = {t:p.infer_chunk(seqs) for t,p in pipes.items()}
    separate_counts = {name:sum(map(len, rows)) for name,rows in calls.items()}
    for rows in calls.values(): rows.clear()
    budgets.clear()
    joint = euk.infer_shared({t:pipes[t] for t in order}, seqs)
    for task in pipes:
        for key in ('scores', 'routes', 'probs', 'labels', 'lengths'):
            np.testing.assert_array_equal(joint[task][key], separate[task][key])
    assert joint['euk']['routes'].tolist() == [0, 1, 2, 0, 2, 1]
    assert joint['fun']['routes'].tolist() == [2, 0, 1, 1, 2, 2]
    assert joint['euk']['labels'].tolist() == [False, False, True, True, False, True]
    assert joint['fun']['labels'].tolist() == [True, True, True, False, False, False]
    assert calls == {'dna': [[0,1,2,3,4,5]], 'nt': ([[1,2,4,5],[0,2,3,4,5]] if order[0]=='euk' else [[0,2,3,4,5],[1,2,4,5]]), 'esm': ([[2,4],[0,4,5]] if order[0]=='euk' else [[0,4,5],[2,4]])}
    assert separate_counts == {'dna': 12, 'nt': 9, 'esm': 5}
    assert budgets == [128000]
    # Chunk-local arrays must not leak into later chunks or conflate equal sequences.
    for rows in calls.values(): rows.clear()
    out=euk.infer_shared(pipes, [seqs[2],seqs[2],seqs[0]])
    assert calls['dna']==[[2,2,0]] and len(out['euk']['scores'])==3


def test_h1_exit_skips_all_language_models(pipelines):
    pipes, calls, budgets = pipelines
    for p in pipes.values():
        p.heads.head_prob=lambda name, matrix: np.full(len(matrix), .99)
    out=euk.infer_shared(pipes,[b'A'*1000])
    assert calls=={'dna':[[0]],'nt':[],'esm':[]}
    assert all(v['routes'].tolist()==[0] for v in out.values())


def test_rejects_unrelated_encoders(pipelines):
    pipes, _, _ = pipelines
    pipes['fun'].encoders=object()
    with pytest.raises(ValueError,match='shared encoder'):
        euk.infer_shared(pipes,[b'A'*1000])


@pytest.mark.parametrize('options', [['all'], ['both'], ['euk,fun'], ['euk','fun'], ['fun,euk']])
def test_cli_dual_task_spellings_use_joint_inference(pipelines, options):
    import csv
    import io
    from eukcontigminer.cli import parse_tasks, write_predictions, metadata
    pipes,calls,_=pipelines
    tasks=parse_tasks(options)
    stream=io.StringIO()
    write_predictions([(str(i),b'A'*(1000+i)) for i in range(6)],pipes,stream,tasks,metadata())
    rows=list(csv.DictReader(io.StringIO(stream.getvalue()),delimiter='\t'))
    assert rows[0]['label_euk']=='other' and rows[0]['label_fun']=='fungi'
    assert len(calls['dna'])==1 and len(calls['nt'])==2
    assert len(calls['esm'])==2


def test_exact_threshold_and_adjacent_scores_keep_strict_boundaries(pipelines):
    pipes,_,_=pipelines
    values=np.array([np.nextafter(.5,0.),.5,np.nextafter(.5,1.)])
    for p in pipes.values():
        p.heads.gates={'H1':[0.,0.],'H2':[0.,0.]}
        p.heads.head_prob=lambda name,matrix:values[matrix[:,0].astype(int)]
    seqs=[b'A'*(1000+i) for i in range(3)]
    expected={t:p.infer_chunk(seqs) for t,p in pipes.items()}
    actual=euk.infer_shared(pipes,seqs)
    for t in pipes:
        for key in ('scores','routes','labels'):
            np.testing.assert_array_equal(actual[t][key],expected[t][key])
        assert actual[t]['routes'].tolist()==[0,2,0]
        assert not actual[t]['labels'][1]


def test_identical_language_model_context_is_encoded_once(pipelines):
    pipes,calls,_=pipelines
    pipes['fun'].heads=pipes['euk'].heads
    euk.infer_shared(pipes,[b'A'*(1000+i) for i in range(6)])
    assert calls['nt']==[[1,2,4,5]]
    assert calls['esm']==[[2,4]]
