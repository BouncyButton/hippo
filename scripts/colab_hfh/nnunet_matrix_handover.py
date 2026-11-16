"""Reuse the verified between-run handover for the remaining nine matrix arms."""
import inspect
from pathlib import Path
from scripts.colab_hfh import nnunet_constraints_handover as previous


def main(root):
    source = inspect.getsource(previous.main)
    source = source.replace('nnunet_long_constraints_v1', 'nnunet_long_matrix_v1')
    source = source.replace("'scripts.colab_hfh.nnunet_long_constraints'",
                            "'scripts.colab_hfh.nnunet_long_matrix'")
    source = source.replace('three_model_recovery_after_constraints.log',
                            'three_model_recovery_after_long_matrix.log')
    ns = dict(previous.__dict__)
    exec(compile(source, __file__, 'exec'), ns)
    ns['main'](root)


if __name__ == '__main__': main(Path('/content/hippo_hfh_baselines_v2'))
