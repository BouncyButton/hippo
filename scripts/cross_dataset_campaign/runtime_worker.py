"""Fresh trainer processes from a CUDA-free, preloaded fork server."""
import importlib
import multiprocessing
import os
from pathlib import Path
import signal
import sys


def execute(module,arguments,cwd,log):
    os.chdir(cwd)
    with open(log,'a',buffering=1) as stream:
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
        sys.stdout=os.fdopen(1,'w',buffering=1,closefd=False)
        sys.stderr=os.fdopen(2,'w',buffering=1,closefd=False)
        import torch
        if torch.cuda.is_initialized() or torch.cuda._is_in_bad_fork():
            raise RuntimeError('Worker inherited initialized CUDA state')
        sys.argv=[module,*arguments]
        importlib.import_module(module).main()


class TrainerProcess:
    def __init__(self,command,cwd,log):
        if command[1]!='-m' or command[2]!='scripts.cross_dataset_campaign.msd_mixed_losses':
            raise ValueError('Only the frozen mixed-loss MSD trainer may use this worker')
        if Path(command[0]).resolve()!=Path(sys.executable).resolve():
            raise ValueError('Fork-server interpreter differs from trainer interpreter')
        multiprocessing.set_forkserver_preload(['scripts.cross_dataset_campaign.preload_msd_runtime'])
        self.process=multiprocessing.get_context('forkserver').Process(target=execute,
            args=(command[2],command[3:],str(cwd),str(log)))
        self.process.start()

    def poll(self):return self.process.exitcode

    @property
    def returncode(self):return self.process.exitcode

    @property
    def pid(self):return self.process.pid

    def send_signal(self,value):
        try:os.kill(self.pid,value)
        except ProcessLookupError:pass

    def wait(self):
        self.process.join();return self.returncode
