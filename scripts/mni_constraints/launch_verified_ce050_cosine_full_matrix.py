"""Launch only the archive-gated half-CE/cosine all-model matrix."""
import subprocess
from scripts.mni_constraints import verified_ce050_cosine_full_matrix as configured
from scripts.mni_constraints import launch_verified_unetr_full_matrix as launcher

assert launcher.study is configured.matrix
assert launcher.study.MODULE=='scripts.mni_constraints.verified_ce050_cosine_full_matrix'

if __name__=='__main__':
    try:launcher.main()
    except subprocess.CalledProcessError as error:
        print(error.stderr)
        raise
