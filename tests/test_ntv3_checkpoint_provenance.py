import pytest
from eukcontigminer.ntv3_runtime import validate_checkpoint_progress, validate_checkpoint_provenance

@pytest.mark.parametrize('field,value', [('epoch', 3), ('step', 32768)])
def test_both_training_schedules_preserve_exact_provenance(field, value):
    binding = {'checkpoint_plan_sha256': 'a'*64, 'checkpoint_'+field: value}
    checkpoint = {'plan_sha256': 'a'*64, field: value}
    validate_checkpoint_provenance(checkpoint, binding)
    for altered in ({**checkpoint, field: value+1}, {**checkpoint, 'plan_sha256': 'b'*64}, {'plan_sha256': 'a'*64}, {**checkpoint, field: float(value)}):
        with pytest.raises(ValueError):
            validate_checkpoint_provenance(altered, binding)

@pytest.mark.parametrize('binding', [{}, {'checkpoint_epoch': 3, 'checkpoint_step': 32768}, {'checkpoint_step': None}, {'checkpoint_step': True}, {'checkpoint_step': 0}, {'checkpoint_step': -1}, {'checkpoint_step': 3.0}, {'checkpoint_step': '32768'}])
def test_missing_ambiguous_or_invalid_progress_fails_closed(binding):
    with pytest.raises(ValueError):
        validate_checkpoint_progress(binding)
