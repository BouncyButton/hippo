"""Scheduling-only continuation: original gain EDGE+DEPTH, seed 0, folds 1/2."""
import os
from pathlib import Path
import signal
import time

from scripts.colab_hfh.medsam_gain_replication_campaign import GainReplication, portable_gpu_sample
from scripts.colab_hfh.medsam_recovery import copy_verified
from scripts.colab_hfh.runtime import read, write


class EdgeDepthContinuation(GainReplication):
    def __init__(self, root):
        super().__init__(root)
        audit = read(self.work / 'edge_depth_resume_audit.json')
        if audit['status'] != 'verified' or audit['folds'] != [1, 2]:
            raise ValueError('Matching checkpoint/protocol audit required')
        if 'L4' not in audit['gpu']:
            raise ValueError('Requested L4 required')
        self.state = read(self.work / 'state_before_edge_depth.json')
        self.state.update(status='starting_edge_depth_seed0', workers=[], error=None,
            continuation_scope='Only gain EDGE+DEPTH, seed0 folds1/2; reuse native controls; stop afterwards',
            requested_workers=1, queued_folds=[1, 2], automatic_expansion=False)
        for fold in (1, 2):
            self.load_control(0, fold)

    def prepare(self, recipe, arm, seed, fold):
        if arm != 'dice_edge_depth' or seed != 0 or fold not in (1, 2):
            raise ValueError('Outside explicit EDGE+DEPTH continuation')
        return super().prepare(recipe, arm, seed, fold)

    def run(self):
        try:
            for fold in (1, 2):
                self.state.update(status='running_edge_depth_seed0', current_pair=dict(seed=0, fold=fold),
                                  queued_folds=[f for f in (1, 2) if f > fold])
                # One fold per phase guarantees one active worker; the next fold waits.
                self.phase({'name': 'gain_edge_depth_continuation'}, 'dice_edge_depth',
                           [(0, fold)], 'results', False)
            self.state.update(status='complete_edge_depth_seed0_folds12', queued_folds=[],
                              next_action='Report paired results; no further runs authorized')
        except SystemExit as error:
            if error.code != 75:
                raise
            self.state['status'] = 'paused'
        except Exception as error:
            self.state.update(status='error', error=str(error)[-3500:])
            raise
        finally:
            for job in self.active.values():
                while job['process'].poll() is None and not (job['output'] / 'supervisor_ready.json').exists():
                    time.sleep(2)
                if job['process'].poll() is None:
                    job['process'].send_signal(signal.SIGUSR1)
            for job in self.active.values():
                job['process'].wait()
            (self.work / 'STOP').touch()
            copy_verified(Path(__file__), self.durable / 'edge_depth_seed0_continuation_driver.py')
            self.publish(portable_gpu_sample(), [], True)


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    EdgeDepthContinuation(Path('/content/hippo_hfh')).run()
