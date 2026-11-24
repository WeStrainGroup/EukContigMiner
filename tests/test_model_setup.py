"""Public option spelling and offline model preparation contracts."""
import hashlib
import json
from pathlib import Path

import pytest

from eukcontigminer import cli, resources, setup_models


@pytest.mark.parametrize('options', [
    ['-o', 'out'], ['--out', 'out'], ['--inp', 'in.fa'], ['-h'],
])
def test_prediction_requires_full_option_names(options):
    with pytest.raises(SystemExit) as error:
        cli.main(['--input', 'input.fa', '--output', 'results', *options])
    assert error.value.code == 2


@pytest.mark.parametrize('options', [['--out', 'models'], ['-h']])
def test_download_requires_full_option_names(options):
    with pytest.raises(SystemExit) as error:
        cli.main(['download-models', *options])
    assert error.value.code == 2


def test_verify_models_accepts_only_full_options():
    with pytest.raises(SystemExit) as error:
        cli.main(['verify-models', '--mod', 'models'])
    assert error.value.code == 2


def test_verify_models_validates_every_file_without_network(tmp_path, monkeypatch, capsys):
    payloads = {'euk/h1.pt': b'head', 'encoders/nt/config.json': b'config'}
    manifest = {'files': {name: {'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data in payloads.items()}}
    monkeypatch.setattr(resources, 'metadata', lambda: manifest)
    def fail_network(*args, **kwargs):
        raise AssertionError('offline verification must not download')
    monkeypatch.setattr(setup_models.urllib.request, 'urlretrieve', fail_network)
    for name, data in payloads.items():
        p=tmp_path/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    cli.main(['verify-models', '--models', str(tmp_path)])
    assert json.loads(capsys.readouterr().out)['files'] == 2
    (tmp_path/'euk/h1.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='Missing or changed model'):
        cli.main(['verify-models', '--models', str(tmp_path)])
    (tmp_path/'euk/h1.pt').unlink()
    with pytest.raises(ValueError, match='Missing or changed model'):
        cli.main(['verify-models', '--models', str(tmp_path)])
