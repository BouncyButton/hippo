"""Run one matrix cell with the established gain, BAND, SDF, EDGE and DEPTH recipes."""
from hashlib import sha256
import math
from pathlib import Path
import signal

from scripts.colab_hfh import medsam_adaptive_band as band
from scripts.colab_hfh import medsam_gain_replication as gain
from scripts.colab_hfh.medsam_constraints_policy import ARMS
from scripts.colab_hfh.medsam_adaptive_band_worker import worker_source


def check_protocol(p):
    gain.check_protocol(p)
    if p['seed'] != 0 or p['fold'] not in range(3) or len(p['requested_arms']) != 1:
        raise ValueError('Expected one seed-0 matrix cell')
    arm, = p['requested_arms']
    if arm not in ARMS or arm in ('dice', 'dice_edge_depth'):
        raise ValueError('Unexpected or completed matrix condition')
    if p['adaptive_band'] != band.POLICY or p['constraint_warmup_updates'] != 20 or p['learning_rate_horizon_updates'] != 200:
        raise ValueError('Frozen policy changed')
    for name, expected in p['matrix_source_hashes'].items():
        if sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed source: '+name)


def install(engine, protocol, output):
    check_protocol(protocol)
    arm, = protocol['requested_arms']
    if 'bands' in arm:
        # Each worker is a dedicated process with one protocol for its lifetime.
        band.check_protocol = check_protocol
        band.install(engine, protocol, output)
    else:
        gain.install(engine, protocol, output)


def install_calibration(engine, protocol, output):
    from scripts.cross_dataset_campaign import foundation as foundation
    check_protocol(protocol)
    band.check_protocol = check_protocol
    band.install_calibration(engine, protocol, output)
    # Add EDGE/DEPTH before the component logging preflight so the preflight
    # tests the objective the worker will actually optimize.
    from scripts.colab_hfh import medsam_augmentation_components as component
    import inspect
    existing = engine.calibrate
    closure = inspect.getclosurevars(existing).nonlocals
    base_calibrate = closure['reference']

    def calibrate(model, criterion, parameters, slabs, current, arm, destination):
        result = base_calibrate(model, criterion, parameters, slabs, current, arm, destination)
        terms = [term for term in ('edge','depth') if term in arm.split('_')]
        if terms:
            from scripts.cross_dataset_campaign.adni_foundation_full_runtime import REFERENCE_CALIBRATE
            with band.preserve_runtime(model, slabs):
                legacy = REFERENCE_CALIBRATE(model, criterion, parameters, slabs, current,
                                            'dice_'+'_'.join(terms), destination)
            anchors = [[c,z] for c,z in slabs.calibration_schedule(current['calibration_slabs_per_volume'])]
            if len(anchors) != 40 or [[r['case'],r['anchor']] for r in legacy['rows']] != anchors:
                raise ValueError('EDGE/DEPTH calibration sample mismatch')
            result['edge_depth_calibration'] = legacy
            result['weights'].update(legacy['weights'])
        return result

    # The component wrapper has a single calibration callable in its closure.
    import types
    def cell(value):
        return (lambda: value).__closure__[0]
    cells = tuple(cell(calibrate) if k == 'reference' else c
                  for k,c in zip(existing.__code__.co_freevars, existing.__closure__))
    engine.calibrate = types.FunctionType(existing.__code__, existing.__globals__, existing.__name__,
                                         existing.__defaults__, cells)


def validate(output, p, arm, history, calibration):
    from scripts.colab_hfh.runtime import read
    if set(k for k,v in calibration['weights'].items() if v) != set(arm.split('_')[1:])-{'bands'}:
        raise ValueError('Initial active constraints differ')
    if 'sdf' in arm:
        c = calibration['sdf_calibration']
        if c['weight'] != calibration['weights']['sdf'] or any(c[k]!=v for k,v in band.calibration_summary(c['rows'],'sdf').items()):
            raise ValueError('SDF calibration mismatch')
    if 'edge' in arm or 'depth' in arm:
        c=calibration['edge_depth_calibration']
        if len(c['rows'])!=40 or {r['case'] for r in c['rows']}!=set(p['train_cases']):
            raise ValueError('EDGE/DEPTH calibration support differs')
        for term,w in c['weights'].items():
            if not 0<w or w!=calibration['weights'][term] or c['statistics'][term]['weighted_p95']>.50000001:
                raise ValueError('EDGE/DEPTH calibrated coefficient differs')
    state = band.empty_state()
    if 'bands' in arm:
        final = read(output/'adaptive_band_state.json')
        probes = {r['update']:r for r in final['probes']}
        assert list(probes)==list(range(0,125,20))
        calibrations = {c['identity']['update']:c for c in final['calibrations']}
    else:
        probes = {}
        final = state
    for i,h in enumerate(history):
        if i in probes:
            probe=probes[i]
            obs=band.summarize_errors(probe['crops'])
            assert {r['case'] for r in obs['crops']}==set(p['train_cases'])
            state,event=band.transition(state,obs,i)
            if event=='activate':
                c=calibrations[i]
                assert c['identity']['update']==i and c['term']=='bands'
                assert all(c[k]==v for k,v in band.calibration_summary(c['rows'],'bands').items())
                state['weight']=c['weight']
                state['calibrations'].append(c)
        ramp=min((i+1)/20,1.)
        value=h['native']+ramp*sum(w*h[k] for k,w in calibration['weights'].items())
        if 'bands' in arm:
            report=dict(active=state['active'],band_ramp=band.band_factor(state,i),band_weight=state['weight'],
                        last_probe=state['last_probe'],activation_update=state['activation_update'])
            assert h['adaptive_band']==report
            value+=report['band_ramp']*state['weight']*h['bands']
        assert h['ramp']==ramp and math.isclose(h['total'],value,rel_tol=2e-5,abs_tol=2e-4)
    assert state==final


if __name__=='__main__':
    signal.signal(signal.SIGUSR1,signal.SIG_IGN)
    source=worker_source().replace('from scripts.colab_hfh.medsam_adaptive_band import install',
                                   'from scripts.colab_hfh.medsam_seed0_matrix_worker import install')
    exec(compile(source,__file__,'exec'),{'__name__':'__main__','__file__':__file__})
