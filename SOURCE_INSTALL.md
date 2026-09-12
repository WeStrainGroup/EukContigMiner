# Installing from a source checkout

For normal use, install the wheel from the matching GitHub release. It includes the trained ECM checkpoint. NTv3 and ESM-C upstream files are obtained separately as described in [NTV3_INSTALL.md](NTV3_INSTALL.md).

The fully trained NTv3 checkpoint is approximately 427 MB and is distributed in the release wheel rather than as a Git object. Cloning the source alone does not supply that checkpoint. Its SHA256 is recorded in `eukcontigminer/model_data/model.json`.

To develop from a source checkout, download the wheel for the same release tag, then run this from the repository root with its path substituted below. This extracts only the trained checkpoint, verifies its expected hash, and refuses to overwrite a different local file.

```python
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

wheel = Path("/path/to/the-matching-release.whl")
root = Path("eukcontigminer/model_data")
config = json.loads((root / "model.json").read_text())
binding = config["model"]["nt_adapter"]["checkpoint"]
assert binding["asset"] == "nt_adapter.pt"
target = root / binding["asset"]

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

with zipfile.ZipFile(wheel) as archive:
    packaged = json.loads(archive.read("eukcontigminer/model_data/model.json"))
    assert packaged["release_version"] == config["release_version"]
    assert packaged["model"]["nt_adapter"]["checkpoint"] == binding
    if target.exists():
        assert digest(target) == binding["sha256"], "Local checkpoint differs"
    else:
        fd, name = tempfile.mkstemp(prefix=".nt_adapter-", dir=root)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as output:
                with archive.open("eukcontigminer/model_data/nt_adapter.pt") as source:
                    shutil.copyfileobj(source, output)
            assert digest(temporary) == binding["sha256"], "Wheel checkpoint differs"
            os.link(temporary, target)  # Fails if another process created the target.
        finally:
            temporary.unlink(missing_ok=True)
print("Trained ECM checkpoint verified:", target)
```

Then install the source with `python -m pip install -e .` and follow the upstream-backbone setup in `NTV3_INSTALL.md`. Keep `nt_adapter.pt` out of Git. The application and model licences continue to apply to source-based installations.
