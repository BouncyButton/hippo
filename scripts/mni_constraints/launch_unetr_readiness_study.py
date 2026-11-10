"""Reuse the verified two-GPU deployment/collection path for readiness trials."""
import subprocess
from scripts.mni_constraints import launch_unetr_delayed_combinations as launcher
from scripts.mni_constraints import unetr_readiness_study as study


if __name__=='__main__':
    for name in ('ROOT','MODULE','PARENT','NAME','verify'):
        setattr(launcher,name,getattr(study,name))
    try:
        launcher.main()
    except subprocess.CalledProcessError as error:
        print(error.stderr)
        raise
