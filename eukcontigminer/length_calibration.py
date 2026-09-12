"""Frozen continuous length decision-score transform; not posterior calibration."""
import numpy as np

def validate_length_calibration(config):
    if config.get('schema') != 'ecm.log_length_decision.v1' or config.get('knots_bp') != [1000,2000,5000,10000,100000]:
        raise ValueError('Unsupported length calibration')
    coefficients=np.asarray(config.get('coefficients'),dtype=float)
    if coefficients.shape!=(5,) or not np.isfinite(coefficients).all():
        raise ValueError('Invalid length calibration coefficients')

def apply_length_calibration(scores,lengths,config):
    validate_length_calibration(config)
    from scipy.special import expit,logit
    scores=np.asarray(scores,dtype=float);lengths=np.asarray(lengths)
    if scores.shape!=lengths.shape or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)) or not np.all(lengths>=1):
        raise ValueError('Invalid length calibration inputs')
    knots=np.log(config['knots_bp']);x=np.log(np.clip(lengths.astype(float),1000,100000))
    basis=np.column_stack([np.interp(x,knots,np.eye(5)[:,k]) for k in range(5)])
    threshold=expit(basis@config['coefficients'])
    result=expit(logit(np.clip(scores,1e-15,1-1e-15))-logit(np.clip(threshold,1e-15,1-1e-15)))
    result=np.where(scores==threshold,.5,result)
    result=np.where((scores>threshold)&(result<=.5),np.nextafter(.5,1.),result)
    result=np.where((scores<threshold)&(result>=.5),np.nextafter(.5,0.),result)
    if not np.isfinite(result).all():raise ValueError('Nonfinite calibrated score')
    return result
