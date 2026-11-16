"""Explicit optimized entry points with the existing matrix protocols and resume checks."""
import runpy
import subprocess
import sys
from unittest.mock import patch

from scripts.colab_hfh import medsam_seed0_matrix_campaign as campaign

WORKER='scripts.colab_hfh.medsam_seed0_matrix_worker'
THIS_MODULE='scripts.colab_hfh.medsam_seed0_accelerated_campaign'


def main():
    if '--arm' in sys.argv:
        from scripts.colab_hfh.medsam_matrix_acceleration import install
        install()
        runpy.run_module(WORKER, run_name='__main__')
        return

    # Use the frozen campaign's scheduling, hash checks and archive format while
    # selecting the accelerated entry point only for the three matrix workers.
    from scripts.colab_hfh.medsam_constraints_drive import client
    client().resolve(campaign.REMOTE + "/runs", create=True)
    original_popen=subprocess.Popen
    def start(cmd,*args,**kwargs):
        if isinstance(cmd,list) and cmd[:3]==[sys.executable,'-m',WORKER]:
            cmd=[sys.executable,'-m',THIS_MODULE,*cmd[3:]]
        return original_popen(cmd,*args,**kwargs)
    with patch.object(subprocess,'Popen',start):
        campaign.run()


if __name__=='__main__':
    main()
