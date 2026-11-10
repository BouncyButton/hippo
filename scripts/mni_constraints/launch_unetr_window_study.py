"""Deploy the frozen finite-window study after the observational diagnosis."""
import subprocess
from scripts.mni_constraints import launch_unetr_delayed_combinations as launcher
from scripts.mni_constraints import unetr_window_study as study

if __name__=='__main__':
    for name in ('ROOT','MODULE','PARENT','NAME','verify'):setattr(launcher,name,getattr(study,name))
    try:launcher.main()
    except subprocess.CalledProcessError as error:
        print(error.stderr)
        raise
