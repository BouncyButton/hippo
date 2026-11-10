"""Deploy the two unchanged UNETR AdamW diagnostic replays."""
import subprocess
from scripts.mni_constraints import launch_unetr_delayed_combinations as launcher
from scripts.mni_constraints import unetr_optimizer_probe_study as study


if __name__=='__main__':
    for name in ('ROOT','MODULE','PARENT','NAME','verify'):
        setattr(launcher,name,getattr(study,name))
    try:launcher.main()
    except subprocess.CalledProcessError as error:
        print(error.stderr)
        raise
