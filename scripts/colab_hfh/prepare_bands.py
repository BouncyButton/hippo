"""Bind BAND experiments to the 36 verified, successful HFH control runs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT/'reports/hfh_colab_exploration_20261111'
ADNI = ROOT/'reports/adni_colab_seeds014_drive_optimized_20261018/package/seed4_bundle/pilot_protocols'
RATES = {'nnunet':.1,'unetr_vit_b16':.0003,'basic_unet16':.0003,'swinunetr':.001}


def build():
    protocols, references = {}, {}
    for archive in sorted((EXPORT/'recipe_analysis').glob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            if 'results/archive_manifest.json' not in z.namelist(): continue
            for name, expected in json.loads(z.read('results/archive_manifest.json')).items():
                assert hashlib.sha256(z.read('results/'+name)).hexdigest()==expected
            p=json.loads(z.read('protocol/protocol.json')); model=p['model_identity']; alias=p['trainer_alias']
            if p['initial_lrs'][alias]!=RATES[model]: continue
            result=json.loads(z.read('results/result.json'))
            key=f"{model}_f{result['fold']}_s{result['seed']}"
            if key in references: continue
            assert result['validation_dice']>=.5 and result['updates']==p['models'][alias]['objectives']['dice']['max_steps']
            assert hashlib.sha256(z.read('protocol/protocol.json')).hexdigest()==json.loads(z.read('results/runtime_recipe.json'))['protocol_sha256']
            references[key]=dict(result=result, initialization=json.loads(z.read('results/initialization.json')),
                configuration=json.loads(z.read('results/config.json')),
                validation=json.loads(z.read('results/validation_latest.json')),
                training=json.loads(z.read('results/training_final.json')),
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                baseline_protocol_sha256=hashlib.sha256(z.read('protocol/protocol.json')).hexdigest(),
                observed_gpu='NVIDIA L4', precision='bf16')
            if model in protocols: continue
            parent=json.loads((ADNI/model/'protocol.json').read_text())
            schedule=deepcopy(parent['models'][alias]['objectives']['dice_bands'])
            assert schedule['max_steps']==result['updates']
            assert schedule['consumed_training_flops']<=p['budget_flops']
            assert schedule['baseline_flops_per_update']==p['models'][alias]['objectives']['dice']['flops_per_update']
            p=deepcopy(p)
            p['models'][alias]['objectives']['dice_bands']=schedule
            p.update(name='HFH_matched_BAND_v1', constraints=True, seeds=[0,1,2], folds=[0,1,2],
                requested_arm_order=['dice_bands'], sdf_policy={'dice':p['sdf_policy']['dice'],'dice_bands':parent['sdf_policy']['dice_bands']},
                calibration_policy='Training-only gradient calibration after frozen native-loss prefix; costs separately reported',
                band_definition='Canonical balanced inner/outer BCE, two six-connected morphological VOXEL steps; no focal or degree weighting',
                pairing='Same initialization, seed, patient split, optimizer, native loss, LR and update allowance as archived successful HFH controls',
                baseline_protocol_sha256=references[key]['baseline_protocol_sha256'],
                band_cost_parent_sha256=hashlib.sha256((ADNI/model/'protocol.json').read_bytes()).hexdigest(),
                endpoint='Final budget; all 36 comparisons retained, no seed or fold rejection')
            protocols[model]={'protocol.json':p,
                'splits_10patients.json':json.loads(z.read('protocol/splits_10patients.json')),
                'nnunet_architecture.json':json.loads(z.read('protocol/nnunet_architecture.json'))}
            # Preserve exact split/architecture bytes: their hashes are scientific bindings.
            protocols[model]['splits_10patients.json']=z.read('protocol/splits_10patients.json').decode()
            protocols[model]['nnunet_architecture.json']=z.read('protocol/nnunet_architecture.json').decode()
    assert len(references)==36 and len(protocols)==4
    payload=dict(version='HFH_BAND_v1', precision='bf16', seeds=[0,1,2],folds=[0,1,2],
        rates=RATES,protocols=protocols,references=references,
        interpretation='Paired exploratory validation comparison; improvement is measured, not a completion requirement')
    destination=EXPORT/'bands_v1_payload.json'
    destination.write_text(json.dumps(payload,indent=2)+'\n')
    print(destination)
    return payload


if __name__=='__main__': build()
