"""One fixed UNETR Dice+CE run using the audited screening/recovery engine."""
import argparse
import json
from pathlib import Path

try:
    import runtime as engine
except ImportError:
    from scripts.colab_geofwi_seed_search import runtime as engine

io = engine.io
MODEL = 'unetr_vit_b16'
RUN_ID = 'GeoFWI3D_unetr_vit_b16_dice_ce_pilot_f0s0'
INITIALIZATION = '1215193efc51fc67a9da9cb1ef0bca0e6e8019d4af8146d9efcb85b94bb051f3'
BASE_VALIDATE = engine.validate_output


def task(seed=0, fold=0, model=MODEL):
    if (seed, fold, model) != (0, 0, MODEL):
        raise ValueError('Pilot permits only UNETR seed 0 fold 0')
    return dict(id=RUN_ID, kind='conventional', dataset='GeoFWI3D', model=MODEL,
                fold=0, seed=0, arm='dice', protocol='unetr_dice_ce_pilot_f0s0')


def validate_output(root, out, t):
    if t != task():
        raise ValueError('Unexpected pilot task')
    recipe = io.read(out/'runtime_recipe.json')
    if recipe['weight_ce'] != 1 or recipe['weight_dice'] != 1 or recipe['augmentation'] != 'none':
        raise ValueError('Pilot loss recipe changed')
    row = BASE_VALIDATE(root, out, t)
    if row['initialization_sha256'] != INITIALIZATION:
        raise ValueError('Initialization differs from archived Dice control')
    return row


def decide(seed, rows):
    if rows:
        raise ValueError('Completed pilot must stop before another task is selected')
    return dict(seed=0, status='pending', next_task=task(), completed_cells=0)


def publish(durable, rows, status=None):
    if len(rows) > 1 or any(r['id'] != RUN_ID for r in rows):
        raise ValueError('Unexpected results in the isolated pilot folder')
    reference = io.read(durable/'dice_control.json')
    value = dict(status='complete' if rows else (status or 'not_started'),
                 scope='One paired loss pilot; not completion of the 3-seed/3-fold baseline',
                 runs=rows, dice_control=reference,
                 final_dice_threshold=0.5, primary_metric='final_budget_validation_dice')
    if rows:
        value.update(final_dice_delta_percentage_points=100*(rows[0]['final_dice']-reference['final_dice']),
                     threshold_passed=rows[0]['final_dice'] > 0.5,
                     next_step='Review this result before authorizing any additional training')
    io.write(durable/'pilot_status.json', value)
    if rows:
        io.write(durable/'PILOT_COMPLETE.json', value)
    print(json.dumps(value, indent=2), flush=True)
    return value


def install():
    # The existing engine retains its budget, pause, recovery and verified-archive logic.
    engine.MODELS = (MODEL,)
    engine.task = task
    engine.decide = decide
    engine.publish = publish
    engine.validate_output = validate_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['verify', 'preflight', 'run', 'report'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--durable', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=2)
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error('Positive session limit required')
    install()
    root = io.activate(args.bundle)
    durable = args.durable.resolve()
    durable.mkdir(parents=True, exist_ok=True)
    engine.verify(root)
    io.copy_verified(root/'dice_control.json', durable/'dice_control.json')
    if args.command == 'verify':
        print('Pilot code, data, splits and frozen recipe verified')
    elif args.command == 'preflight':
        if io.environment() != io.read(root/'control_environment.json'):
            raise ValueError('GPU/software differs from archived control. Use its pinned environment and A100-SXM4-80GB before paired training.')
        engine.preflight(root, durable)
    elif args.command == 'run':
        engine.run(root, durable, args.hours, max_runs=1)
    else:
        publish(durable, engine.load_rows(root, durable))


if __name__ == '__main__':
    main()
