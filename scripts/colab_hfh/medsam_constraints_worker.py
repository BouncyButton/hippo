"""Arm-aware bridge to the audited MedSAM training/evaluation/recovery runner."""
from pathlib import Path
import signal


def worker_source():
    source = (Path(__file__).parents[1]/'colab_mni_medsam3/optimized.py').read_text()
    replacements = (
        ('from scripts.colab_medsam3.runtime import RecoveryWriter, copy_verified, restore',
         'from scripts.colab_hfh.medsam_recovery import RecoveryWriter, restore'),
        ('from scripts.colab_mni_medsam3.train import SupportSlabs',
         'from scripts.colab_hfh.medsam_constraints_support import PairedSlabs as SupportSlabs, bind_learning_rates'),
        ('from scripts.colab_mni_medsam3.data import Volumes', 'from scripts.colab_hfh.data import Volumes'),
        ('    args = parser.parse_args()',
         "    parser.add_argument('--arm', required=True)\n    args = parser.parse_args()\n"
         "    from scripts.colab_hfh.medsam_constraints_policy import ARMS\n"
         "    if args.arm not in ARMS or (args.arm == 'dice' and not read(args.protocol)['constraint_recipe'].get('native_control')): raise ValueError('Invalid constraint arm')\n"
         "    from scripts.colab_hfh.medsam_constraints_deterministic import install as install_numerics\n"
         "    install_numerics(read(args.protocol))\n"
         "    bind_learning_rates(original, read(args.protocol))\n"
         "    from scripts.colab_hfh.medsam_constraints_loss_variants import install as install_loss\n"
         "    install_loss(original, read(args.protocol))\n"
         "    from scripts.colab_hfh.medsam_constraints_schedule import install as install_schedule\n"
         "    install_schedule(original, read(args.protocol))\n"
         "    from scripts.colab_hfh.medsam_constraints_resume import install as install_resume\n"
         "    install_resume(original)"),
        ("expected = {**read(args.protocol), 'arm':'dice'}", "expected = {**read(args.protocol), 'arm':args.arm}"),
        ("    checkpoint = output/'checkpoint_latest.pt'",
         "    checkpoint = output/'checkpoint_latest.pt'\n"
         "    if os.environ.get('HFH_REQUIRE_COMPLETED_F0S0') == '1' and output.name == 'strict_native__dice__f0_s0':\n"
         "        if not checkpoint.exists() or torch.load(checkpoint,map_location='cpu',weights_only=False)['step'] != 200:\n"
         "            raise RuntimeError('Completed fold0 seed0 recovery required; retraining is forbidden')"),
        ('    original.calibrate = matched.calibrate',
         '    from scripts.cross_dataset_campaign import adni_foundation_full_runtime as full_runtime\n'
         '    original.calibrate = full_runtime.calibrate'),
        ("'--base-weights',str(args.base_weights),'--arm','dice'", "'--base-weights',str(args.base_weights),'--arm',args.arm"),
        ("training='Unchanged native-loss trainer; exact CPU transform cache'",
         "training='Native plus calibrated constraints; exact paired samples/LR; 200 updates'"),
        ('                copy_verified(p,durable/p.name)', '                pass  # Controller commits one verified result ZIP.'),
        ('            copy_verified(checkpoint,durable/checkpoint.name)', '            pass  # Immutable RecoveryWriter already preserved recovery.'),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Audited worker changed; review adaptation: '+old)
        source = source.replace(old, new)
    return source


def main():
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    namespace = {'__name__': '__main__', '__file__': __file__}
    exec(compile(worker_source(), __file__, 'exec'), namespace)


if __name__ == '__main__':
    main()
