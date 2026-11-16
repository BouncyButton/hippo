"""CPU smoke check with the actual native deep-supervised criterion and trainer."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import torch.nn.functional as F
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from scripts.cross_dataset_campaign import conventional as exp
from scripts.adni_constraints import objectives as binary
from scripts.colab_hfh.residual_band import policy_dict
from scripts.colab_hfh.residual_band_worker import install,calibration_case,bind_trace,worker_source
from scripts.colab_hfh.residual_band_campaign import campaign_source
from scripts.colab_hfh.residual_band_archive import validate_trace


def main():
    torch.set_num_threads(2)
    p=json.loads(Path('reports/hfh_colab_exploration_20261111/nnunet_conditional_band_v1/payload.json').read_text())['protocols']['nnunet']['protocol.json']
    assert p['residual_band_policy']==policy_dict()
    install(exp.trainer,p,'dice_bands')
    compile(worker_source(),'generated_worker','exec')
    compile(campaign_source(),'generated_campaign','exec')
    criterion=DeepSupervisionWrapper(DC_and_CE_loss(
        dict(batch_dice=False,smooth=1e-5,do_bg=False),{},weight_ce=1,weight_dice=1,
        ignore_label=None,dice_class=MemoryEfficientSoftDiceLoss),[4/7,2/7,1/7,0])
    labels=torch.zeros((1,1,64,64,64),dtype=torch.long);labels[:,:,20:44,22:42,20:44]=1
    z=torch.zeros((1,2,64,64,64));z[:,1]=torch.where(labels[:,0]>0,2.,-2.)
    z[:,1,20,25:29,25:29]=-2
    outputs=[F.interpolate(z,size=(n,n,n),mode='nearest').requires_grad_() for n in (64,32,16,8)]
    native=binary.training_loss(outputs,labels,'dice',{},1.,criterion=criterion)
    with TemporaryDirectory() as folder:
        root=Path(folder);(root/'budget_ledger.json').write_text(json.dumps(dict(charged_attempts=46)))
        wrapped=bind_trace(binary.training_loss,root)
        loss=wrapped(outputs,labels,'dice_bands',{'bands':.1},1.,criterion=criterion)
        assert float(loss)>float(native)
        loss.backward()
        assert all(o.grad is not None and torch.isfinite(o.grad).all() for o in outputs[:3])
        report=calibration_case(outputs,labels,'dice_bands',criterion=criterion)
        assert report['gradient_names']==['native','bands']
        row=json.loads((root/'residual_band_trace.jsonl').read_text())
        assert row['charged_attempt']==46 and row['selected_errors']==16
        # Native-only calls must not make a conditional decision.
        wrapped(outputs,labels,'dice',{},1.,criterion=criterion)
        assert len((root/'residual_band_trace.jsonl').read_text().splitlines())==1
        (root/'residual_band_trace.jsonl').write_text(''.join(json.dumps(dict(row,charged_attempt=i))+'\n' for i in range(46,91)))
        (root/'config.json').write_text(json.dumps(dict(train_cases=['train'],validation_cases=['heldout'])))
        (root/'residual_band_spatial_final.json').write_text(json.dumps(dict(cases=[
            dict(split='train',case_name='train',annotation_sha256=row['annotation_sha256']),
            dict(split='validation',case_name='heldout',annotation_sha256='different')])) )
        assert validate_trace(root,p)['active_attempts']==45
        bad=dict(row,charged_attempt=46,annotation_sha256='different')
        lines=(root/'residual_band_trace.jsonl').read_text().splitlines();lines[0]=json.dumps(bad)
        (root/'residual_band_trace.jsonl').write_text('\n'.join(lines)+'\n')
        try:validate_trace(root,p)
        except ValueError as e:assert 'Non-training' in str(e)
        else:raise AssertionError('Held-out annotation was accepted')
    print('PASS: actual trainer install, native deep supervision, auxiliary gradient, envelope calibration, training-only trace and archive gate')


if __name__=='__main__':main()
