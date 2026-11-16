"""Same optimization budget and loss; change only ramp and add diagnostics."""
import hashlib
from pathlib import Path
from scripts.colab_hfh.residual_band_protocol import payload as old_payload
from scripts.colab_hfh.residual_band_v2 import VERSION

FILES=('residual_band_v2.py','residual_band_v2_worker.py','residual_band_v2_protocol.py',
       'residual_band_v2_archive.py','residual_band_v2_campaign.py','residual_band_v2_pilot.py')


def payload(base,plan):
    result=old_payload(base,plan)
    p=result['protocols']['nnunet']['protocol.json']
    p.update(name=VERSION,conditional_ramp='min(active_update_count/9,1); inactive calls do not advance; global per run',
        constraint_definition=p['constraint_definition']+'; activation-count ramp; read-only actual-gradient diagnostics',
        residual_band_v2_source_sha256={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in FILES},
        diagnostics=dict(full_resolution_gradients='Every eligible update; actual native, weighted FN, FP, total auxiliary and combined gradients',
            parameter_probes='First active update, 60, 75, 90; maximum four; three extra backwards and one post-step forward each',
            optimizer_updates=0,parameter_probe_max_flops=16*355220395319,
            logit_probe_max_flops=45*8192*64**3,
            accounting='Read-only diagnostics separately bounded outside the unchanged optimization cap; no additional optimizer steps',
            recovery='Ramp count reconstructed from completed-update trace; duplicate/replayed charged attempts rejected'))
    # A few scalar ramp-counter operations fit within the already conservative
    # conditional arithmetic allowance. All diagnostic branches are read-only.
    result.update(version=VERSION,folds=[0],seeds=[0,1,2],interpretation='Three-seed diagnostic rerun only; no automatic fold expansion')
    return result
