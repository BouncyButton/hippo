"""Concurrency decisions and completed-run preservation checks."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from scripts.colab_hfh.medsam_constraints_capacity import Capacity
from scripts.colab_hfh.medsam_constraints_parallel import Campaign


def gpu(used=11,slots=3):
    return dict(total_gib=39.5,used_gib=used,host_free_gib=60,cpu_slots=slots,
                utilization=70,processes={1:10,2:10,3:10})


def workers(n,step,phase='training'):
    return [dict(name=str(i),pid=i,step=step,phase=phase,peak_gib=10) for i in range(1,n+1)]


class CapacityTests(unittest.TestCase):
    def window(self,tuner,n,start,steps):
        tuner.observe(start,gpu(10*n),workers(n,20))
        tuner.observe(start+180,gpu(10*n),workers(n,20+steps))

    def test_only_safe_workers_admitted(self):
        c=Capacity();c.target=3
        self.assertTrue(c.admission(gpu(0),[]))
        self.assertFalse(c.admission(gpu(30),workers(2,20)))
        self.assertFalse(c.admission(gpu(10),workers(1,0,'starting')))
        self.assertFalse(c.admission({**gpu(10),'host_free_gib':15},workers(1,20)))

    def test_benchmark_all_feasible_counts_and_select_best(self):
        c=Capacity()
        self.window(c,1,0,30)
        self.assertEqual(c.target,2)
        self.window(c,2,200,24);self.assertIsNone(c.decision)
        self.window(c,2,400,24);self.assertEqual(c.target,3)
        self.window(c,3,600,15);self.window(c,3,800,15)
        self.assertEqual(c.decision['workers'],2)
        self.assertAlmostEqual(c.decision['rates'][1],10)
        self.assertAlmostEqual(c.decision['rates'][2],16)
        self.assertAlmostEqual(c.decision['rates'][3],15)

    def test_no_gain_returns_to_one_without_discarding_workers(self):
        c=Capacity(maximum=2)
        self.window(c,1,0,30)
        self.window(c,2,200,15);self.window(c,2,400,15)
        self.assertEqual(c.target,1)
        self.assertFalse(c.admission(gpu(20),workers(2,150)))

    def test_mixed_phases_never_count_as_training_speedup(self):
        c=Capacity();c.target=2
        w=workers(2,20);w[0]['phase']='evaluation'
        c.observe(0,gpu(),w);c.observe(600,gpu(),w)
        self.assertFalse(c.measurements)

    def test_phase_or_task_changes_reset_window(self):
        c=Capacity();c.observe(0,gpu(),workers(1,20))
        w=workers(1,100);w[0]['name']='different'
        c.observe(200,gpu(),w)
        self.assertFalse(c.measurements)

    def test_peak_reservation_does_not_drop_when_worker_finishes(self):
        c=Capacity();w=workers(1,20);w[0]['peak_gib']=15
        c.observe(0,gpu(),w);before=c.allowance
        c.observe(5,gpu(),workers(1,20))
        self.assertEqual(c.allowance,before)


class PreservationTests(unittest.TestCase):
    def test_completed_archives_are_collected_without_launching_workers(self):
        with TemporaryDirectory() as directory:
            campaign=Campaign.__new__(Campaign)
            campaign.maximum=0;campaign.active={};campaign.stopping=False
            campaign.work=Path(directory);campaign.durable=Path(directory)/'remote'
            campaign.state={'controls':[]}
            campaign.prepare=Mock(side_effect=lambda recipe,arm,seed,fold:dict(seed=seed,fold=fold))
            campaign.existing=Mock(side_effect=lambda job:dict(seed=job['seed'],fold=job['fold'],validation_dice=.8))
            campaign.launch=Mock(side_effect=AssertionError('Completed run must not launch'))
            campaign.publish=Mock()
            with patch('scripts.colab_hfh.medsam_constraints_parallel.gpu_sample',return_value=gpu()),\
                    patch('scripts.colab_hfh.medsam_constraints_parallel.copy_verified'):
                rows=campaign.phase({'name':'strict_native'},'dice',[(0,0),(0,1)],'controls',True)
            self.assertEqual(len(rows),2)
            campaign.launch.assert_not_called()

    def test_archived_negative_band_stops_recipe_without_starting_pending_pairs(self):
        with TemporaryDirectory() as directory:
            campaign=Campaign.__new__(Campaign)
            campaign.maximum=0;campaign.active={};campaign.stopping=False
            campaign.work=Path(directory);campaign.durable=Path(directory)/'remote'
            campaign.state={'results':[]}
            out=Path(directory)/'pending';out.mkdir()
            campaign.prepare=Mock(side_effect=lambda recipe,arm,seed,fold:dict(seed=seed,fold=fold,output=out))
            campaign.existing=Mock(side_effect=lambda job:dict(seed=0,fold=0,validation_dice=.8,delta_dice_pp=-.1) if job['fold']==0 else None)
            campaign.launch=Mock(side_effect=AssertionError('Rejected recipe must not launch'))
            rows=campaign.phase({'name':'band'},'dice_bands',[(0,0),(0,1)],'results',True)
            self.assertEqual(len(rows),1)
            campaign.launch.assert_not_called()


if __name__=='__main__':unittest.main()
