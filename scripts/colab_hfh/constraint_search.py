"""Sequential recipe screening and a separately frozen complete HFH matrix."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.colab_hfh.bands_campaign import read, write
from scripts.colab_hfh.constraint_drive import Drive
from scripts.colab_hfh.constraint_schedule import make_payload, make_schedule
from scripts.colab_hfh.constraint_calibration import components


def qualifies(rows):
    return (len(rows) == 9
            and {(r['fold'], r['seed']) for r in rows} == {(f,s) for f in range(3) for s in range(3)}
            and all(r['delta_dice_pp'] >= 1. and r['constraint_dice'] >= .5 for r in rows))


def near_threshold_followup(rows):
    """Collect all nine for promising near misses without declaring them passed."""
    return (len(rows)==1 and .9<=rows[0]['delta_dice_pp']<1.
            and rows[0]['constraint_dice']>=.5)


def prefer_mean(candidate, incumbent):
    """Rank complete qualifying recipes on the same nine fixed comparisons."""
    return qualifies(candidate) and (not qualifies(incumbent)
        or sum(r['delta_dice_pp'] for r in candidate) > sum(r['delta_dice_pp'] for r in incumbent))


def recipes(model, plan, arm):
    known = plan['verified_band_recipes'].get(model)
    if known:
        prefix, ramp = known['native_prefix'], known['ramp']
    else:
        prefix, ramp = (22,9) if model == 'nnunet' else (0,50)
    targets = (.1, .03, .3)
    recipes = [dict(prefix=prefix, ramp=ramp, target=target, cap=.5 if target <= .1 else 1.)
               for target in targets]
    if model in ('nnunet', 'swinunetr'):
        recipes.append(dict(prefix=0 if model == 'nnunet' else 50,
                            ramp=45 if model == 'nnunet' else 50, target=.1, cap=.5))
        if model=='nnunet' and 'sdf' in arm:
            recipes[-1]['sdf_ramp']=9
    if model == 'nnunet' and 'bands' in arm and any(term in arm for term in ('depth','sdf')):
        # Inner BAND recovered misses but raised false positives; the measured
        # depth pilot reduced false positives while increasing misses. Test the
        # complementary effects together, with an explicit joint gradient cap.
        recipes.extend(dict(prefix=22,ramp=9,target=.3,cap=.75,inner_weight=inner)
                       for inner in (.9,.75))
    if 'bands' in arm and len(components(arm))>1:
        for main_target, auxiliary_target in ((.1,.025),(.3,.05)):
            spec=dict(prefix=prefix,ramp=ramp,target=main_target,cap=.5 if main_target==.1 else .75,
                      component_targets={term:main_target if term=='bands' else auxiliary_target
                                         for term in components(arm)})
            if model=='nnunet' and main_target==.3 and any(term in arm for term in ('depth','sdf')):
                spec['inner_weight']=.9
            recipes.append(spec)
    if model=='nnunet' and 'edge' in arm:
        # Choose onset from arithmetic, never validation: retain the native
        # 90 updates while reserving a complete nine-update auxiliary ramp.
        costs=plan['cost_references'][model]['canonical_costs']
        native_updates=plan['cost_references'][model]['native_updates']
        late_prefix=next(start for start in range(native_updates//2,native_updates-9)
            if make_schedule(costs,arm,budget=plan['budget_flops_per_run'],prefix=start,ramp=9)['max_steps']==native_updates)
        for main_target in (.1,.3):
            recipes.append(dict(prefix=late_prefix,ramp=9,target=main_target,
                cap=.5 if main_target==.1 else .75,
                component_targets={term:main_target if term=='bands' or 'bands' not in arm
                                   else main_target/4 for term in components(arm)}))
    if 'bands' in arm and len(components(arm))>1:
        # The audited combined target shrank BAND by 9-14% in tested edge/depth
        # arms. Separate that effect from adding a small auxiliary component.
        # The actual combined/ramp-subset gradient remains explicitly bounded.
        anchors = (.1,) if known else ((.3,.4) if model=='swinunetr' else (.3,.3))
        for index,anchor in enumerate(anchors):
            spec=dict(prefix=prefix,ramp=ramp,target=1.5*anchor if known else anchor+.1,
                      cap=.5 if known else .75,
                      component_targets={term:anchor if term=='bands' else .01
                                         for term in components(arm)})
            if model=='nnunet' and index==1:
                spec['inner_weight']=.75
            recipes.append(spec)
    if arm=='dice_sdf' and model in ('nnunet','swinunetr'):
        # Isolate calibration time: keep the original SDF onset/ramp and native
        # updates, but measure its coefficient after foreground learning.
        late_prefix,unused_early_ramp,sdf_ramp=(45,4,9) if model=='nnunet' else (103,10,50)
        for target in (.1,.3):
            recipes.append(dict(prefix=late_prefix,ramp=unused_early_ramp,sdf_ramp=sdf_ramp,
                                target=target,cap=.5 if target==.1 else 1.))
        if model=='swinunetr':
            # Measured final SDF strength fell to 2-21% of its calibrated
            # ratio. Test a separate training-only online strength recipe;
            # its detached probe is charged inside the optimization cap.
            for target in (.1,.3):
                recipes.append(dict(prefix=late_prefix,ramp=unused_early_ramp,sdf_ramp=sdf_ramp,
                    target=target,cap=.5 if target==.1 else 1.,online_sdf_ratio=target))
    if arm == 'dice_bands' and model in ('nnunet', 'swinunetr'):
        # Fixed side emphasis repeatedly traded false positives for misses.
        # Test training-case difficulty weighting without tuning each held-out
        # pair, changing the native schedule, or replacing prior candidates.
        recipes.extend(dict(prefix=prefix, ramp=ramp, target=target, cap=1. if target==.3 else .5,
                            adaptive_sides=True) for target in (.3, .1))
    recipes.extend(spec for spec in diagnosis_recipes(model, plan, arm) if spec not in recipes)
    return recipes


def diagnosis_recipes(model, plan, arm):
    """Fixed follow-ups to the archived HFH/MNI error analysis; no pair tuning."""
    if model == 'nnunet':
        # Append only: existing candidate IDs/protocols remain immutable.
        # HFH misses exceed false positives, foreground hard-Dice emerges near
        # mid-budget, and early SDF calibration loses strength by the endpoint.
        # These fixed, validation-informed exploratory hypotheses are shared by
        # all nine pairs. Calibration itself still uses training crops only.
        costs = plan['cost_references'][model]['canonical_costs']
        native_updates = plan['cost_references'][model]['native_updates']
        late_prefix = next(start for start in range(native_updates//2, native_updates-9)
            if make_schedule(costs, arm, budget=plan['budget_flops_per_run'],
                             prefix=start, ramp=9)['max_steps'] == native_updates)
        if arm == 'dice_bands':
            additions = [dict(prefix=start, ramp=9, target=.3, cap=.75, inner_weight=.9)
                         for start in (22, late_prefix)]
        elif 'bands' in arm:
            additions = [dict(prefix=late_prefix, ramp=9, target=.4, cap=.75,
                inner_weight=.9, component_targets={term: .3 if term == 'bands' else .01
                                                    for term in components(arm)})]
        else:
            # Depth/edge can suppress true foreground; first isolate a weak
            # later correction without changing native CE, LR or momentum.
            additions = [dict(prefix=late_prefix, ramp=9, target=.03, cap=.15)]
        if arm in ('dice_bands', 'dice_bands_depth'):
            # Fixed 0.9 inner emphasis recovered misses but harmed cases that
            # already oversegmented. Test a milder late 0.75/0.25 balance on
            # every pair; never choose sides from held-out case outcomes.
            moderate = dict(prefix=late_prefix, ramp=9, target=.3, cap=.75,
                            inner_weight=.75)
            if arm == 'dice_bands_depth':
                moderate.update(target=.4, component_targets={'bands':.3,'depth':.01})
            additions.append(moderate)
        return additions
    return []


def run(args):
    if getattr(args, 'optimize_mean', False):
        from scripts.colab_hfh.constraint_mean_search import run_mean
        return run_mean(args)
    root = args.root
    work = root/'constraint_search';work.mkdir(exist_ok=True)
    plan = read(root/'constraint_matrix_plan.json')
    base = read(root/'bands_v1_payload.json')
    for name,expected in plan['calibration_implementation']['source_sha256'].items():
        if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Constraint asset source differs from the declared plan: '+name)
    drive = Drive('results_constraint_search_v1')
    path = work/'search_status.json'
    if not path.exists():
        matches = [r for r in json.loads(drive.call('lsjson', drive.folder, '--files-only', '--hash'))
                   if r['Name'] == path.name]
        if len(matches) > 1:
            raise ValueError('Ambiguous saved search')
        if matches:
            drive.call('copyto', drive.path(path.name), str(path), '--checksum')
            if hashlib.md5(path.read_bytes()).hexdigest() != matches[0]['Hashes'].get('MD5', matches[0]['Hashes'].get('md5')):
                raise ValueError('Search recovery checksum mismatch')
    state = read(path) if path.exists() else dict(status='prepared', attempts={}, selected={},
        criterion='All nine fixed pairs >=+1.0 Dice pp and >=50% Dice',
        interpretation='Adaptive exploratory validation selection; no independent generalization claim',
        reused_conditions=54, total_matrix_conditions=504)
    def persist():
        write(path, state);drive.upload(path)
    def publish_report():
        from scripts.colab_hfh.constraint_report import write_report
        summary, matrix_path, summary_path = write_report(root)
        drive.upload(matrix_path);drive.upload(summary_path)
        print('FINAL-MATRIX COVERAGE',summary['verified'],'of',summary['conditions'],flush=True)
    deadline = time.time()+args.hours*3600
    # After the repeated FP/FN tradeoffs in boundary/depth arms, test the
    # distance-field objective before more combinations of those same terms.
    order = ['dice_depth', 'dice_sdf', 'dice_bands_depth', 'dice_bands_edge', 'dice_bands_sdf', 'dice_edge_depth',
             'dice_sdf_depth', 'dice_bands_edge_depth', 'dice_bands_sdf_depth',
             'dice_edge_sdf_depth', 'dice_bands_edge_sdf', 'dice_bands_edge_sdf_depth', 'dice_bands']
    models = ['nnunet','swinunetr','basic_unet16','unetr_vit_b16']
    requested = getattr(args,'models',None)
    if requested:
        if len(set(requested))!=len(requested) or not set(requested)<=set(models):
            raise ValueError('Unknown or duplicate model scope')
        models = [model for model in models if model in requested]
    optimize_mean = args.phase=='screen' and getattr(args,'optimize_mean',False)
    queue = [(model,arm) for arm in order for model in models
             if not (arm == 'dice_bands' and model in plan['verified_band_recipes'])]
    queue.sort(key=lambda pair: pair[0] not in ('nnunet','swinunetr'))
    if args.pilot:
        queue = [('nnunet','dice_depth')]
    state.update(status='running', phase=args.phase, model_scope=models,
                 ranking='mean_delta_among_all_nine_qualified' if optimize_mean else 'first_all_nine_qualified')
    persist()
    for model, arm in queue:
        key = model+'__'+arm
        selected = state['selected'].get(key)
        if args.phase=='screen' and selected and selected['qualification']!='all_nine_passed':
            # A newly appended hypothesis may resume screening; already frozen
            # full matrices are never silently reselected.
            completed_full=any(r.get('final_matrix_frozen') for r in state['attempts'].values()
                               if r['model']==model and r['objective']==arm)
            pending=any(f'{key}__r{i:02d}' not in state['attempts'] for i in range(len(recipes(model,plan,arm)))) or any(
                near_threshold_followup(r.get('results',[])) for r in state['attempts'].values()
                if r['model']==model and r['objective']==arm)
            if pending and not completed_full:
                state['selected'].pop(key);selected=None
        if args.phase == 'matrix' and selected is None:
            raise ValueError('Freeze a recipe for every arm before full-matrix execution: '+key)
        specs = recipes(model, plan, arm)
        def candidate_indexes():
            if args.phase == 'matrix':
                yield selected['candidate'];return
            # Keep stable IDs; test smaller added terms before equal-strength
            # combinations, followed by the earlier complementary-error arms.
            yielded=set()
            if not args.pilot:
                for i,spec in enumerate(specs):
                    if 'component_targets' in spec:
                        yielded.add(i);yield i
                for i,spec in enumerate(specs):
                    if 'inner_weight' in spec and i not in yielded:
                        yielded.add(i);yield i
            yield 0
            yielded.add(0)
            if args.pilot:
                return
            initial = state['attempts'].get(key+'__r00',{}).get('results',[])
            # Keep stable candidate definitions. A positive but insufficient
            # initial result prioritizes stronger supervision; a regression
            # prioritizes gentler supervision. Never tune per validation pair.
            positive = bool(initial and initial[0]['delta_dice_pp'] > 0)
            for i in ([2,1] if positive else [1,2])+list(range(3,len(specs))):
                if i not in yielded:
                    yielded.add(i);yield i
        indexes = candidate_indexes()
        for index in indexes:
            name = f'{key}__r{index:02d}'
            record = state['attempts'].setdefault(name, dict(model=model, objective=arm,
                candidate=index, specification=specs[index], status='pending'))
            if record['specification'] != specs[index]:
                raise ValueError('Frozen candidate definition changed')
            near = args.phase=='screen' and not args.pilot and (near_threshold_followup(record.get('results',[]))
                    or (record.get('near_threshold_followup') and not record.get('full_matrix_complete')))
            if args.phase == 'screen' and ((selected and not optimize_mean) or (record['status'] in ('rejected','qualified') and not near)):
                continue
            if args.phase == 'matrix' and record.get('full_matrix_complete'):
                if not record.get('final_matrix_frozen'):
                    record['final_matrix_frozen']=True;persist()
                continue
            if time.time() >= deadline or (work/'STOP').exists():
                state['status']='paused';persist();publish_report();return
            payload = make_payload(base, model, arm, plan['cost_references'][model]['canonical_costs'],
                                   **specs[index], screening=args.phase == 'screen' and not near)
            payload_path = work/(name+'_payload.json');write(payload_path,payload)
            trial = work/name
            record['status']='running';state['current']=name;persist()
            if near:
                record['near_threshold_followup']=True
                print('NEAR-THRESHOLD FOLLOW-UP',name,'all nine retained; +1pp qualification unchanged',flush=True)
            print('CONSTRAINT RECIPE START',name,specs[index],flush=True)
            try:
                subprocess.run([sys.executable,'-m','scripts.colab_hfh.constraint_campaign',
                    '--root',str(root),'--payload',str(payload_path),'--work',str(trial),
                    '--folder','results_constraints_'+name,'--hours',str(max(.05,(deadline-time.time())/3600))],check=True)
            except subprocess.CalledProcessError:
                record['status']='runtime_error';state['status']='runtime_error';persist();raise
            result = read(trial/'campaign_status.json')
            if result['status'] not in ('rejected_pair','complete_requested_matrix'):
                state['status']='paused';persist();publish_report();return
            rows = result['results']
            if args.phase=='screen' and not args.pilot and not near and near_threshold_followup(rows):
                # The first comparison is already archived. Reuse it and collect
                # the other eight without another validation rejection gate.
                record.update(status='near_threshold_pending',results=rows,near_threshold_followup=True)
                persist()
                if time.time()>=deadline or (work/'STOP').exists():
                    state['status']='paused';persist();publish_report();return
                payload=make_payload(base,model,arm,plan['cost_references'][model]['canonical_costs'],
                                     **specs[index],screening=False)
                write(payload_path,payload)
                print('NEAR-THRESHOLD FOLLOW-UP',name,'all nine retained; +1pp qualification unchanged',flush=True)
                try:
                    subprocess.run([sys.executable,'-m','scripts.colab_hfh.constraint_campaign',
                        '--root',str(root),'--payload',str(payload_path),'--work',str(trial),
                        '--folder','results_constraints_'+name,'--hours',str(max(.05,(deadline-time.time())/3600))],check=True)
                except subprocess.CalledProcessError:
                    record['status']='runtime_error';state['status']='runtime_error';persist();raise
                result=read(trial/'campaign_status.json')
                if result['status']!='complete_requested_matrix':
                    record['results']=result.get('results',rows)
                    state['status']='paused';persist();publish_report();return
                rows=result['results']
            passed = qualifies(rows)
            record.update(status='qualified' if passed else 'rejected',results=rows,
                          full_matrix_complete=len(rows)==9,
                          final_matrix_frozen=args.phase=='matrix' and len(rows)==9)
            print('CONSTRAINT RECIPE RESULT',name,record['status'],'pairs',len(rows),
                  'minimum delta',min(r['delta_dice_pp'] for r in rows),flush=True)
            if passed:
                incumbent = (state['attempts'].get(f'{key}__r{selected["candidate"]:02d}',{}).get('results',[])
                             if selected else [])
                if not optimize_mean or prefer_mean(rows,incumbent):
                    selected = state['selected'][key] = dict(candidate=index, qualification='all_nine_passed',
                        specification=specs[index], mean_delta_pp=sum(r['delta_dice_pp'] for r in rows)/9)
            persist()
            if len(rows)==9 and selected:
                publish_report()
            if selected and not optimize_mean:
                break
        if not args.pilot and args.phase == 'screen' and key not in state['selected']:
            attempted = [r for r in state['attempts'].values()
                         if r['model']==model and r['objective']==arm and r.get('results')]
            if len(attempted) != len(specs):
                raise ValueError('Cannot select from incomplete screening')
            # Every candidate starts with the SAME f0s1 pair; do not compare
            # averages from unequal subsets. This fallback is not a winner.
            consistent_near=[r for r in attempted if len(r['results'])==9
                             and min(x['delta_dice_pp'] for x in r['results'])>=.9
                             and min(x['constraint_dice'] for x in r['results'])>=.5]
            best = (max(consistent_near,key=lambda r:min(x['delta_dice_pp'] for x in r['results']))
                    if consistent_near else max(attempted, key=lambda r: r['results'][0]['delta_dice_pp']))
            state['selected'][key] = dict(candidate=best['candidate'], specification=best['specification'],
                qualification=('not_qualified_consistent_near_threshold_all_nine' if consistent_near
                               else 'not_qualified_best_common_first_pair_for_complete_reporting'))
            persist()
            if len(best['results'])==9:
                publish_report()
    state.update(status='pilot_complete' if args.pilot else ('screen_complete' if args.phase=='screen' else 'matrix_complete'))
    state.pop('current',None);persist();print('CONSTRAINT SEARCH STATUS',state['status'],flush=True)
    publish_report()


def run_all(args):
    """Screen, freeze recipes, then complete the matrix under one deadline."""
    deadline = time.time()+args.hours*3600
    for phase in ('screen', 'matrix'):
        remaining = deadline-time.time()
        if remaining <= 0 or (args.root/'constraint_search/STOP').exists():
            return
        current = argparse.Namespace(**{**vars(args), 'phase': phase, 'hours': remaining/3600})
        run(current)
        state = read(args.root/'constraint_search/search_status.json')
        if state['status'] != phase+'_complete':
            return


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--phase',choices=('screen','matrix','all'),default='screen')
    parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--models',nargs='+',choices=('nnunet','swinunetr','basic_unet16','unetr_vit_b16'))
    parser.add_argument('--optimize-mean',action='store_true')
    parser.add_argument('--objectives',nargs='+',help='Explicit objectives for the mean-recovery phase')
    args=parser.parse_args()
    if args.objectives and not args.optimize_mean:
        parser.error('Objective scope is supported only with --optimize-mean')
    if args.pilot and args.phase=='all':
        parser.error('Pilot mode cannot automatically expand to the matrix')
    # Same controller lock as strict BAND search; never two experiment owners.
    with (args.root/'bands_recipe_search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run_all(args) if args.phase=='all' else run(args)
