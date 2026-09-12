# Install the NTv3 100M edition

Use Python 3.10–3.12 and install the wheel attached to the selected [EukContigMiner release](https://github.com/WeStrainGroup/EukContigMiner/releases). The wheel includes ECM's trained models; upstream backbones are downloaded once, separately.

```bash
python -m pip install /path/to/eukcontigminer-VERSION-py3-none-any.whl
```

Accept the applicable model terms for [NTv3 100M](https://huggingface.co/InstaDeepAI/NTv3_100M_pre) and [ESM-C 300M](https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12). If access is gated, authenticate the download environment with your approved Hugging Face account (`huggingface-cli login`). Do not put access tokens in scripts or command-line arguments.

Run the following after installation, with network access enabled. The installed model configuration supplies the exact NTv3 revision and required files, so these cannot drift independently of the release.

```python
import hashlib
import json
from importlib.resources import files
from pathlib import Path
from huggingface_hub import snapshot_download

config = json.loads(
    files("eukcontigminer").joinpath("model_data/model.json").read_text()
)
nt = config["model"]["nt_adapter"]
assert nt["schema"] == "ecm.ntv3.runtime.ieee.v1"
assert nt["backbone_model_id"] == "InstaDeepAI/NTv3_100M_pre"
snapshot_download("EvolutionaryScale/esmc-300m-2024-12")
folder = Path(snapshot_download(
    nt["backbone_model_id"],
    revision=nt["backbone_revision"],
    local_dir="models/ntv3_100m",
    allow_patterns=sorted(nt["source_files_sha256"]),
))
for name, expected in nt["source_files_sha256"].items():
    digest = hashlib.sha256()
    with (folder / name).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"Model file differs from the release: {name}")
print("NTv3 files verified:", folder.resolve())
```

For inference, use the same Hugging Face cache that contains ESM-C:

```bash
export EUKCONTIGMINER_NTV3_DIR="$PWD/models/ntv3_100m"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
eukcontigminer input.fna.gz -o predictions.tsv --device cuda:0
```

For CPU, use `--device cpu --cpu-threads 4`. `--device auto` selects an available GPU or CPU. Inference requires no reference database or runtime network access. The NT branch verifies its upstream files and trained checkpoint before loading them.

Output columns are `contig_id`, `length_bp`, `p_euk`, and `label`. The model configuration contains the frozen global threshold. Every nonempty legal contig receives a finite score in [0, 1]; only scores strictly greater than the threshold are labelled Eukaryota. Equality is Other. Organelles are positive; there is no Unknown class. All lengths are scored; formal performance evaluation begins at 1,000 bp. A decision score is not a guarantee of posterior probability at an arbitrary sample prevalence.

## Model terms

Application code is MIT. NTv3 and its adapted derivative are governed by the included InstaDeep Open Model Licence 1.0 (April 2025), including its noncommercial restrictions. The adapted ECM checkpoint was modified by supervised training. ESM-C has separate upstream terms. See `NTV3-MODEL-LICENSE.md` and `THIRD_PARTY_NOTICES.md` in the release; the application's MIT licence does not replace the model terms.
