"""Scheduling-only fold-2 check; preserve the paused gain campaign and recipes."""
import os
from pathlib import Path
import signal
import time

from scripts.colab_hfh.medsam_gain_replication_campaign import GainReplication, portable_gpu_sample
from scripts.colab_hfh.medsam_gain_replication import CAMPAIGN
from scripts.colab_hfh.medsam_recovery import copy_verified
from scripts.colab_hfh.runtime import read, write


class Fold2Check(GainReplication):
    def __init__(self, root):
        prior = read(root / CAMPAIGN / 'campaign_status.json')
        if prior['status'] != 'paused' or prior.get('workers'):
            raise ValueError('Requires the checkpoint-safe paused campaign')
        super().__init__(root)
        self.state = prior
        self.state.update(status='starting_fold2_check', error=None,
                          current_pair=dict(seed=0, fold=2),
                          continuation_scope='Only fold2 seed0 native/BAND/EDGE+DEPTH; unchanged recipes; stop afterwards')

    def prepare(self, recipe, arm, seed, fold):
        if (seed, fold) != (0, 2):
            raise ValueError('This continuation may run only fold2 seed0')
        return super().prepare(recipe, arm, seed, fold)

    def run(self):
        try:
            self.state['status'] = 'running_native'
            self.phase({'name': 'gain_native'}, 'dice', [(0, 2)], 'controls', False)
            self.load_control(0, 2)
            self.state['status'] = 'running_constraints'
            self.phase_pair(0, 2)
            controls = [r for r in self.state['controls'] if (r['seed'], r['fold']) == (0, 2)]
            results = [r for r in self.state['results'] if (r['seed'], r['fold']) == (0, 2)]
            if len(controls) != 1 or sorted(r['arm'] for r in results) != ['dice_bands', 'dice_edge_depth']:
                raise ValueError('Incomplete fold2 comparison')
            self.state.update(status='complete_fold2_seed0_check',
                              next_action='Compare folds before any recipe change or further seed expansion')
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
            write(self.work / 'fold2_continuation_scope.json', dict(
                seed=0, fold=2, updates=125, arms=['dice', 'dice_bands', 'dice_edge_depth'],
                training_changes=False, automatic_expansion=False))
            copy_verified(Path(__file__), self.durable / 'fold2_continuation_driver.py')
            self.publish(portable_gpu_sample(), [], True)


if __name__ == '__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    Fold2Check(Path('/content/hippo_hfh')).run()
