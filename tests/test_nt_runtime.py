import copy
import pytest
from eukcontigminer.deployment import load_deployment_parameters
from eukcontigminer.nt_runtime import validate_binding,window_spans


def test_window_coordinates_preserve_reverse_complement_at_continuous_lengths():
    for length in list(range(1,10002))+[49999,50000,50001,99999,100000,100001,1000001]:
        spans=window_spans(length)
        assert 1<=len(spans)<=3
        assert len(set(spans))==len(spans)
        assert all(0<=left<right<=length and right-left<=2000 for left,right in spans)
        assert set(spans)=={(length-right,length-left) for left,right in spans}
        if length<=2000:assert spans==[(0,length)]
        else:assert (0,2000) in spans and (length-2000,length) in spans


@pytest.mark.parametrize('key,value',[('alpha',1.),('rank',16),('window_rule','first_only'),('backbone_revision','other'),('forward_reverse_complement','none'),('window_aggregation','max_logits')])
def test_config_cannot_silently_change_frozen_nt_inference(key,value):
    binding=copy.deepcopy(load_deployment_parameters().config['model']['nt_adapter'])
    binding[key]=value
    with pytest.raises(ValueError,match='NT(v3| adapter).*differs'):validate_binding(binding)


def test_empty_nt_window_is_rejected():
    with pytest.raises(ValueError,match='nonempty'):window_spans(0)
