"""Deploy the subfield-sensitive original-loss study only after its parent gate."""
import subprocess
from scripts.mni_constraints import launch_unetr_delayed_combinations as launcher
from scripts.mni_constraints import unetr_subfield_constraints_study as study

if __name__=='__main__':
    for name in ('ROOT','MODULE','PARENT','NAME','verify'):setattr(launcher,name,getattr(study,name))
    try:launcher.main()
    except subprocess.CalledProcessError as error:
        print(error.stderr)
        raise
