"""Complete the long seed-zero canonical matrix, reusing completed controls."""
import inspect
from pathlib import Path

from scripts.colab_hfh import nnunet_long_constraints as previous

NAME = 'nnunet_long_matrix_v1'
ARMS = ('dice_bands', 'dice_sdf', 'dice_bands_sdf', 'dice_bands_edge',
        'dice_bands_edge_sdf', 'dice_bands_depth', 'dice_bands_edge_depth',
        'dice_bands_sdf_depth', 'dice_bands_edge_sdf_depth')
FILES = previous.FILES + ('nnunet_long_matrix.py',)


def namespace():
    # Keep the original archived pilot source intact. Bind its tested controller
    # to a new explicit scope in an isolated module namespace.
    ns = dict(previous.__dict__, NAME=NAME, ARMS=ARMS, FILES=FILES)
    pack_source = inspect.getsource(previous.pack_for)
    old = 'Canonical physical non-BAND constraints'
    if pack_source.count(old) != 1:
        raise ValueError('Previous protocol builder changed')
    pack_source = pack_source.replace(old,
        'Canonical full-matrix constraints (standard BAND, not residual/balanced BAND)')
    exec(compile(pack_source, __file__, 'exec'), ns)
    run_source = inspect.getsource(previous.run)
    old = 'complete_four_arms_no_expansion'
    if run_source.count(old) != 1:
        raise ValueError('Previous controller changed')
    run_source = run_source.replace(old, 'complete_nine_remaining_arms_no_expansion')
    exec(compile(run_source, __file__, 'exec'), ns)
    return ns


def pack_for(base, plan, arm):
    return namespace()['pack_for'](base, plan, arm)


if __name__ == '__main__':
    namespace()['run'](Path('/content/hippo_hfh_baselines_v2'))
