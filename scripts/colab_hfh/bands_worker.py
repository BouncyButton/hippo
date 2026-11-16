"""Run the frozen HFH worker with only the declared BAND objective enabled."""
from pathlib import Path


def worker_source():
    source=Path(__file__).with_name('baseline.py').read_text()
    edits=[
        ("    exp.base.PreparedPatients = PreparedPatients", "    if p.get('band_reset_momentum', False) or p.get('band_release_updates', 0) or 'band_final_inner_weight' in p or p['models'][p['trainer_alias']]['objectives']['dice_bands']['prefix_updates'] == 0:\n        from scripts.colab_hfh.bands_variants import install_training_transition\n        install_training_transition(exp.trainer)\n    exp.base.PreparedPatients = PreparedPatients"),
        ("    binary.bind()", "    if p.get('band_inner_weight', .5) != .5 or p.get('band_focal_gamma', 0) or p.get('band_steps', 2) != 2 or 'band_final_inner_weight' in p:\n        from scripts.colab_hfh.bands_variants import install\n        install(p)\n    binary.bind()"),
        ("p['trainer_alias'], 'dice', 4, True", "p['trainer_alias'], 'dice_bands', 4, True"),
        ("exp.configure(p, 'dice')", "exp.configure(p, 'dice_bands')"),
        ("policy=p['sdf_policy']['dice']", "policy=p['sdf_policy']['dice_bands']"),
        ("        return model\n    exp.base.build_model = build", "        reference=read(args.protocol_dir/'baseline_references.json')[f'{p[\"model_identity\"]}_f{args.fold}_s{args.seed}']\n"
         "        if exp.trainer.model_digest(model) != reference['initialization']['model_state_sha256']:\n"
         "            raise ValueError('BAND initialization differs from matched baseline; no training charged')\n"
         "        return model\n    exp.base.build_model = build"),
        ("loss=p['training_loss'], augmentation='none'", "loss=p['training_loss'], objective='dice_bands', band_definition=p['band_definition'], augmentation='none'"),
    ]
    for before,after in edits:
        if source.count(before)!=1: raise ValueError('HFH worker source changed; review BAND integration')
        source=source.replace(before,after)
    return source


if __name__=='__main__':
    exec(compile(worker_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
