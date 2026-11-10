"""Conditional follow-up: original subfield-sensitive losses without union bands."""
from scripts.mni_constraints import unetr_window_study as engine

ROOT,CAP,PAIRS,PILOTS=engine.ROOT,engine.CAP,engine.PAIRS,engine.PILOTS
NAME='mni_unetr_subfield_losses53t_20261108'
MODULE='scripts.mni_constraints.unetr_subfield_constraints_study'
PARENT='mni_unetr_window53t_20261108'
CORE=engine.CORE
EXPECT_ZERO_FINAL=False
CANDIDATES=tuple(dict(key=key,arm=arm,target=target,taper_start=408,stop=409)
    for key,arm,target in (
        ('sdf_t05','dice_sdf',.05),('depth_t05','dice_depth',.05),
        ('sdf_depth_t05','dice_sdf_depth',.05),('sdf_depth_t20','dice_sdf_depth',.20)))
KEYS=tuple(c['key'] for c in CANDIDATES)
BASE_PROTOCOL=engine.protocol


def protocol(old,profile,candidate):
    p=BASE_PROTOCOL(old,profile,candidate)
    assert p['auxiliary_envelope']==[1.]*409
    p.update(timing_policy='Fresh training-only calibration after102 updates,33-update ramp; original per-class SDF starts nearest5/9 cap. Auxiliary envelope remains1 throughout. No union bands/edge in these arms.',
        decision_rule='Compare original SDF/depth separately and jointly, with fixed5% targets and one20% joint candidate. Same candidate across all seeds/folds. Reject after any nonpositive final paired gain; all9 required.',
        composition_policy='Original SDF and native-depth-weighted CE definitions unchanged; remove foreground-union terms, not replace them with class-aware bands.')
    return p


def bind():
    for key in ('NAME','MODULE','PARENT','CANDIDATES','KEYS','EXPECT_ZERO_FINAL'):
        setattr(engine,key,globals()[key])
    engine.PRIOR_RESULT='window_results.json';engine.PRIOR_OUTCOME='exhausted'
    engine.DECISION_FILE='subfield_constraints_decision.json';engine.protocol=protocol


def tasks():
    bind();return engine.tasks()


def prepare(target=None):
    bind();return engine.prepare(target)


def verify(stage):
    bind();return engine.verify(stage)


def main():
    bind();engine.main()


if __name__=='__main__':main()
