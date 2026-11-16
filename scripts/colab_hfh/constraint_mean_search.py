"""Complete matched nine-pair recipes; freeze each arm at the user's mean target.

This changes selection, not the scientific protocol of any resumed trial.
Historical single-pair rejection decisions remain in the audit history.
"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

from scripts.colab_hfh.bands_campaign import read, write
from scripts.colab_hfh.constraint_drive import Drive
from scripts.colab_hfh.constraint_schedule import make_payload

POLICY = dict(version='mean_one_pp_freeze_v1', minimum_mean_delta_pp=1.,
    required_pairs=9, minimum_final_dice=.5,
    individual_regressions='Allowed; report every pair and the worst case. No numeric regression bound specified.',
    freeze='No new trials for a model/objective once a complete recipe meets the mean target.',
    interpretation='Adaptive exploratory validation selection; no independent generalization claim.')
PAIRS = {(f, s) for f in range(3) for s in range(3)}


def complete(rows):
    return len(rows) == 9 and {(r['fold'], r['seed']) for r in rows} == PAIRS


def qualifies(rows):
    return (complete(rows) and not any(r.get('control_kind') for r in rows)
        and all(math.isfinite(r['delta_dice_pp']) and math.isfinite(r['constraint_dice'])
                and r['constraint_dice'] >= .5 for r in rows)
        and sum(r['delta_dice_pp'] for r in rows)/9 >= 1.)


def summary(rows):
    return dict(pairs=len(rows), complete=complete(rows), qualifies=qualifies(rows),
        mean_delta_pp=sum(r['delta_dice_pp'] for r in rows)/9 if complete(rows) else None,
        worst_delta_pp=min((r['delta_dice_pp'] for r in rows), default=None),
        positive_pairs=sum(r['delta_dice_pp'] > 0 for r in rows),
        pairs_below_one_pp=sum(r['delta_dice_pp'] < 1 for r in rows))


def freeze_qualified(state, models):
    """Reuse already complete evidence, including recipes rejected by the old gate."""
    for name, record in state['attempts'].items():
        if record['model'] not in models or not qualifies(record.get('results', [])):
            continue
        key = record['model']+'__'+record['objective']
        selected = state['selected'].get(key)
        if selected and selected.get('frozen_under_policy') == POLICY['version']:
            continue
        candidates = [(n, r) for n, r in state['attempts'].items()
                      if r['model'] == record['model'] and r['objective'] == record['objective']
                      and qualifies(r.get('results', []))]
        winner, best = max(candidates, key=lambda item: sum(r['delta_dice_pp'] for r in item[1]['results']))
        state['selected'][key] = dict(candidate=best['candidate'], specification=best['specification'],
            recipe=winner, qualification='mean_target_met_all_nine', frozen_under_policy=POLICY['version'],
            **summary(best['results']))


def candidate_order(state, model, arm, specs, preferred_specs=()):
    """Finish active work, then diagnosis-led trials, then common-pilot rank."""
    key = model+'__'+arm
    def priority(index):
        r = state['attempts'].get(f'{key}__r{index:02d}', {})
        rows = r.get('results', [])
        pilot = next((x['delta_dice_pp'] for x in rows if (x['fold'], x['seed']) == (0, 1)), None)
        return (r.get('status') != 'running', specs[index] not in preferred_specs,
                pilot is None, -pilot if pilot is not None else 0., index)
    return [i for i in sorted(range(len(specs)), key=priority)
            if not complete(state['attempts'].get(f'{key}__r{i:02d}', {}).get('results', []))]


def resume_queue(state, queue):
    """Resume the current arm, or advance after a verified completed candidate."""
    current = state['attempts'].get(state.get('current'), {})
    key = state.get('next_arm_key')
    advance = False
    if current:
        key = current['model']+'__'+current['objective']
        advance = complete(current.get('results', []))
    keys = [m+'__'+a for m,a in queue]
    if key not in keys:
        return queue
    index = (keys.index(key)+int(advance)) % len(queue)
    return queue[index:]+queue[:index]


def prioritize_band_followup(state, queue, specs):
    """Honor the BAND priority without interrupting active or repeating completed work."""
    key = ('nnunet', 'dice_bands')
    current = state['attempts'].get(state.get('current'), {})
    if key not in queue or (current.get('status') == 'running' and not complete(current.get('results', []))):
        return queue
    if not complete(state['attempts'].get('nnunet__dice_depth__r04', {}).get('results', [])):
        return queue
    target = dict(prefix=22, ramp=9, target=.3, cap=.75, inner_weight=.9)
    if target not in specs:
        raise ValueError('Declared BAND follow-up specification missing')
    name = f'nnunet__dice_bands__r{specs.index(target):02d}'
    if complete(state['attempts'].get(name, {}).get('results', [])):
        # After its matched BAND+depth counterpart finishes, test the already
        # declared moderate BAND-only recipe before more secondary combinations.
        # This changes queue order only, never candidate IDs or protocols.
        moderate = dict(prefix=45, ramp=9, target=.3, cap=.75, inner_weight=.75)
        if (not complete(state['attempts'].get('nnunet__dice_bands_depth__r11', {}).get('results', []))
                or moderate not in specs):
            return queue
        name = f'nnunet__dice_bands__r{specs.index(moderate):02d}'
        if complete(state['attempts'].get(name, {}).get('results', [])):
            return queue
    # Historical s03 had only four observations under a different archive path.
    # It motivates priority, not reused data or qualification. Run current r06
    # independently on all nine; never pool its results with legacy s03.
    return [key] + [item for item in queue if item != key]


def run_mean(args):
    from scripts.colab_hfh.constraint_search import recipes, diagnosis_recipes
    from scripts.colab_hfh.constraint_report import write_report
    root = args.root
    work = root/'constraint_search'; work.mkdir(exist_ok=True)
    plan, base = read(root/'constraint_matrix_plan.json'), read(root/'bands_v1_payload.json')
    for name, expected in plan['calibration_implementation']['source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != expected:
            raise ValueError('Constraint asset source differs from declared plan: '+name)
    models = getattr(args, 'models', None) or ['nnunet', 'swinunetr']
    if len(set(models)) != len(models) or not set(models) <= {'nnunet', 'swinunetr'}:
        raise ValueError('Mean recovery is scoped to nnU-Net and SwinUNETR only')
    models = [m for m in ('nnunet', 'swinunetr') if m in models]
    drive = Drive('results_constraint_search_v1')
    path = work/'search_status.json'
    if not path.exists():
        matches = [r for r in json.loads(drive.call('lsjson', drive.folder, '--files-only', '--hash'))
                   if r['Name'] == path.name]
        if len(matches) != 1:
            raise ValueError('Expected exactly one historical search to resume')
        drive.call('copyto', drive.path(path.name), str(path), '--checksum')
        if hashlib.md5(path.read_bytes()).hexdigest() != matches[0]['Hashes'].get('MD5', matches[0]['Hashes'].get('md5')):
            raise ValueError('Search recovery checksum mismatch')
    state = read(path)
    if state.get('evaluation_policy') != POLICY:
        state.setdefault('selection_policy_history', []).append(dict(
            criterion=state.get('criterion'), selected=state['selected'].copy(),
            changed_at=time.time(), reason='User revised target to mean >=1pp, permits small regressions, freezes passing arms.'))
    state.update(evaluation_policy=POLICY, criterion='Mean >=+1 Dice pp across all nine fixed pairs; all Dice >=50%',
                 model_scope=models, ranking='finish active candidate; diagnosis-led fixed trials; then common-pilot priority; freeze mean-qualified arms',
                 status='running', phase='mean_recovery')
    freeze_qualified(state, models)
    def persist():
        write(path, state); drive.upload(path)
    def report():
        s, matrix_path, summary_path = write_report(root)
        sync = getattr(drive, 'upload_report', drive.upload)
        pending = []
        for report_path in (matrix_path, summary_path):
            if sync(report_path) is False:
                pending.append(report_path.name)
        state['report_sync_pending'] = pending
        # These files are rebuildable from already verified archives. Record a
        # deferred report locally; the next mandatory state write also carries it.
        write(path, state)
        print('MATRIX COVERAGE', s['verified'], '/', s['conditions'], flush=True)
    def stop(status):
        state.update(status=status); state.pop('current', None); persist(); report()
    # One completion per unresolved arm per round; nnU-Net receives priority.
    arms = ['dice_bands_sdf', 'dice_bands', 'dice_bands_edge_sdf_depth', 'dice_bands_depth',
            'dice_depth', 'dice_sdf_depth', 'dice_bands_sdf_depth', 'dice_bands_edge_sdf',
            'dice_edge_sdf_depth', 'dice_bands_edge_depth', 'dice_bands_edge', 'dice_edge_depth', 'dice_sdf']
    requested = getattr(args, 'objectives', None)
    if requested is not None:
        if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(arms):
            raise ValueError('Invalid mean-recovery objective scope')
        arms = list(requested)
    # User excluded further standalone SDF tuning; preserve historical evidence.
    arms = [arm for arm in arms if arm != 'dice_sdf']
    if not arms:
        raise ValueError('Standalone SDF tuning is excluded from the active scope')
    state.update(objective_scope=arms, excluded_objectives=['dice_sdf'])
    queue = [(m, a) for m in models for a in arms]
    queue = resume_queue(state, queue)
    if 'nnunet' in models:
        queue = prioritize_band_followup(state, queue, recipes('nnunet', plan, 'dice_bands'))
    deadline = time.time()+args.hours*3600
    persist(); report()
    print('MEAN TARGET POLICY: >=+1pp across nine; retain regressions; freeze passing arms', flush=True)
    for key, selected in state['selected'].items():
        if selected.get('frozen_under_policy') == POLICY['version']:
            print('FROZEN', key, selected['recipe'], 'mean', selected['mean_delta_pp'],
                  'worst', selected['worst_delta_pp'], flush=True)
    while True:
        launched = False
        for model, arm in queue:
            key = model+'__'+arm
            selected = state['selected'].get(key, {})
            if selected.get('frozen_under_policy') == POLICY['version']:
                continue
            specs = recipes(model, plan, arm)
            indexes = candidate_order(state, model, arm, specs, diagnosis_recipes(model, plan, arm))
            if not indexes:
                continue
            if time.time() >= deadline or (work/'STOP').exists():
                stop('paused'); return
            index = indexes[0]; name = f'{key}__r{index:02d}'
            record = state['attempts'].setdefault(name, dict(model=model, objective=arm,
                candidate=index, specification=specs[index], status='pending'))
            if record['specification'] != specs[index]:
                raise ValueError('Frozen scientific candidate definition changed')
            if not record.get('mean_completion_started'):
                record['previous_selection_decision'] = dict(status=record['status'],
                    pairs=len(record.get('results', [])), policy=state['selection_policy_history'][-1]['criterion'])
            payload = make_payload(base, model, arm, plan['cost_references'][model]['canonical_costs'],
                                   **specs[index], screening=False)
            payload['evaluation_policy'] = POLICY
            payload_path = work/(name+'_payload.json'); write(payload_path, payload)
            trial = work/name
            record.update(status='running', mean_completion_started=True)
            state['current'] = name; persist()
            print('MEAN COMPLETION START', name, 'saved pairs', len(record.get('results', [])), flush=True)
            try:
                subprocess.run([sys.executable, '-m', 'scripts.colab_hfh.constraint_campaign',
                    '--root', str(root), '--payload', str(payload_path), '--work', str(trial),
                    '--folder', 'results_constraints_'+name,
                    '--hours', str(max(.01, (deadline-time.time())/3600))], check=True)
            except (subprocess.CalledProcessError, OSError):
                record['status']='runtime_error'; state['status']='runtime_error'; persist(); raise
            result = read(trial/'campaign_status.json')
            rows = result.get('results', [])
            record['results'] = rows
            if result['status'] != 'complete_requested_matrix':
                stop('paused'); return
            if not complete(rows):
                raise ValueError('Campaign did not return exactly nine unique fixed pairs')
            record.update(status='qualified_mean' if qualifies(rows) else 'below_mean_target',
                          full_matrix_complete=True, mean_evaluation=summary(rows))
            freeze_qualified(state, models)
            if state['selected'].get(key, {}).get('frozen_under_policy') != POLICY['version']:
                completed = [(n, r) for n, r in state['attempts'].items()
                             if r['model']==model and r['objective']==arm and complete(r.get('results', []))]
                best_name, best = max(completed, key=lambda item: sum(x['delta_dice_pp'] for x in item[1]['results']))
                state['selected'][key] = dict(candidate=best['candidate'], specification=best['specification'],
                    recipe=best_name, qualification='below_mean_target_complete_nine', **summary(best['results']))
            print('MEAN COMPLETION RESULT', name, summary(rows), flush=True)
            next_model, next_arm = queue[(queue.index((model, arm))+1) % len(queue)]
            state['next_arm_key'] = next_model+'__'+next_arm
            persist(); report(); launched = True
        if not launched:
            state['unresolved_arms'] = [m+'__'+a for m,a in queue
                if state['selected'].get(m+'__'+a, {}).get('frozen_under_policy') != POLICY['version']]
            stop('needs_next_hypothesis' if state['unresolved_arms'] else 'all_target_arms_mean_qualified')
            return
