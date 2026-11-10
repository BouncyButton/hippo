from scripts.mni_constraints import launch_nnunet_swin_refine2030_round3 as launcher
from scripts.mni_constraints import nnunet_swin_refine2030_round4 as study

launcher.study=study

if __name__=='__main__':
    try:launcher.main()
    except launcher.subprocess.CalledProcessError as error:
        print(error.output,error.stderr,flush=True);raise
