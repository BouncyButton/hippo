"""Evaluate an explicitly abandoned Dice run, archive it, then retire its artifacts."""
import os
from pathlib import Path
import time
import traceback
from copy import deepcopy
from scripts.cross_dataset_campaign.common import read, write, digest, install_signals, stopping
from scripts.cross_dataset_campaign.storage import retire


def finish(stage, plan):
    retirement = plan['retirement']
    output = stage / 'runs' / retirement['output']
    if (output / 'cleanup.json').exists() and read(output / 'cleanup.json')['status'] == 'complete':
        return True
    install_signals()
    os.environ['HIPPO_CONTROL'] = str(stage / 'control.json')
    status = dict(status='evaluating_interrupted_run', active=output.name, completed=0,
                  total=len(plan['tasks']), job_id=os.environ.get('SLURM_JOB_ID'))
    write(stage / 'status.json', status)
    try:
        if not (output / 'archive_manifest.json').exists():
            import torch
            from scripts.unetr_subfields import core
            from scripts.unetr_subfields.contracts import DATASETS
            torch.set_num_threads(4)
            device = torch.device('cuda')
            assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            parent = Path(retirement['parent_stage'])
            protocol_path = parent / 'protocols' / retirement['dataset'] / 'protocol.json'
            protocol, config = read(protocol_path), read(output / 'config.json')
            if digest(protocol_path) != config['protocol_sha256']:
                raise ValueError('Stopped-run protocol changed')
            checkpoint = torch.load(output / 'checkpoint_latest.pt', map_location='cpu', weights_only=False)
            ledger = read(output / 'budget_ledger.json')
            if checkpoint['configuration'] != config or checkpoint['charged_attempts'] > ledger['charged_attempts']:
                raise ValueError('Stopped checkpoint/ledger mismatch')
            bridge = deepcopy(protocol)
            if bridge['requested_arm_order'] == ['dice_ce']:
                bridge['requested_arm_order'] = ['dice']
                bridge['models']['unet']['objectives'] = {'dice': protocol['models']['unet']['objectives']['dice_ce']}
            core.bind(bridge)
            spec = DATASETS[retirement['dataset']]
            model = core.build_model(len(spec['classes']), spec['spatial_size']).to(device)
            frame = core.base._load_pkl_dataframe(Path(protocol['data']))
            train = core.PreparedSubfields(frame, config['train_cases'], spec)
            validation = core.PreparedSubfields(frame, config['validation_cases'], spec)
            model.load_state_dict(checkpoint['model'])
            latest = core.evaluate(model, validation, device, True, spec['classes'])
            write(output / 'validation_interrupted.json', latest)
            write(output / 'training_interrupted.json', core.evaluate(model, train, device, True, spec['classes']))
            best = torch.load(output / 'checkpoint_best.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(best['model'])
            write(output / 'validation_best_poststop.json', core.evaluate(model, validation, device, True, spec['classes']))
            write(output / 'interrupted_protocol.json', protocol)
            write(output / 'abandonment.json', dict(status='abandoned_by_user', original_run=retirement['original_run'],
                reason=retirement.get('reason', 'User stopped Dice-only seed search and requested a fresh Dice+CE search.'),
                completed_updates=checkpoint['step'], charged_updates=ledger['charged_attempts'],
                consumed_training_flops=ledger['training_flops'], budget_flops=config['budget_flops'],
                additional_optimizer_updates=0, validation_dice=latest['summary']['macro_hard_dice'],
                evaluation_diagnostics='Training and validation Dice, TP/FP/FN and confusion matrices; best and interrupted checkpoint evaluated.',
                excluded_from_complete_seed_results=True, ap_swaps_applicable=False,
                original_source_manifest_sha256=digest(parent / 'manifest.json')))
            del model, checkpoint, best
            torch.cuda.empty_cache()
            metadata, bulky = {}, []
            for path in sorted(output.iterdir()):
                if not path.is_file() or path.is_symlink(): raise ValueError('Unexpected retirement artifact')
                info = dict(bytes=path.stat().st_size, sha256=digest(path))
                if path.suffix in ('.pt', '.tmp') or path.name.endswith('.nii.gz'):
                    bulky.append(dict(name=path.name, **info))
                elif path.suffix in ('.json', '.csv', '.jsonl', '.log', '.lock', '.png'):
                    metadata[path.name] = info
                else: raise ValueError('Unclassified artifact: ' + path.name)
            write(output / 'archive_manifest.json', dict(status='ready',
                task=dict(id=output.name, kind='abandoned', dataset=retirement['dataset']),
                metadata=metadata, deletion_candidates=bulky, created_at=time.time(),
                abandonment_sha256=digest(output / 'abandonment.json')))
        status['status'] = 'waiting_for_verified_archive'
        write(stage / 'status.json', status)
        started = time.monotonic()
        while not (output / 'archive_acknowledgment.json').exists():
            if stopping() or time.monotonic() - started > plan['archive_wait_timeout_seconds']:
                status['status'] = 'paused' if read(stage / 'control.json')['pause'] else 'waiting_for_archive_offline'
                write(stage / 'status.json', status)
                return False
            time.sleep(5)
        retire(output)
        write(stage / 'retirement_complete.json', dict(run=output.name, cleanup=read(output / 'cleanup.json')))
        return not stopping()
    except Exception:
        write(stage / 'failure.json', dict(active=output.name, traceback=traceback.format_exc()))
        status['status'] = 'failed'
        write(stage / 'status.json', status)
        raise
