"""Verify two-volume support and prevent a premature or duplicate handoff."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign import sam3_two_volume_job as job
from scripts.cross_dataset_campaign.common import read, write, digest


class HandoffTests(unittest.TestCase):
    def test_submit_reuses_owned_job(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            write(stage/'submissions/123.json', dict(job_id='123'))
            with patch.object(job.controller, 'queue', return_value=[['123', 'PENDING', 'sam3']]), \
                 patch.object(job.chain, 'submit') as submit:
                self.assertEqual(job.submit(stage), dict(job_id='123'))
                submit.assert_not_called()

    def test_completion_requires_archived_source_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory); run = stage/'runs/one'
            write(stage/'manifest.json', {})
            write(stage/'control.json', dict(pause=False))
            write(stage/'status.json', dict(status='complete', completed=1))
            write(run/'archive_manifest.json', {})
            write(run/'completion.json', dict(status='complete'))
            write(run/'cleanup.json', dict(status='pending'))
            ack = dict(status='verified', source_verified=True,
                source_manifest_sha256=digest(stage/'manifest.json'),
                manifest_sha256=digest(run/'archive_manifest.json'))
            write(run/'archive_acknowledgment.json', ack)
            plan = dict(tasks=[dict(id='one')])
            with self.assertRaisesRegex(ValueError, 'cleanup incomplete'):
                job.predecessor_complete(stage, plan)
            write(run/'cleanup.json', dict(status='complete'))
            self.assertTrue(job.predecessor_complete(stage, plan))
            write(run/'archive_acknowledgment.json', {**ack, 'source_manifest_sha256': 'wrong'})
            with self.assertRaises(ValueError): job.predecessor_complete(stage, plan)
            write(stage/'control.json', dict(pause=True))
            self.assertFalse(job.predecessor_complete(stage, plan))

    def run_chain(self, completed):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory); stage = root/'new'; previous = root/'previous'
            for path in (stage, previous): write(path/'control.json', dict(pause=False))
            write(previous/'status.json', dict(status='waiting_for_archive_offline'))
            stack.enter_context(patch.dict('os.environ', {'SLURM_JOB_ID': '123'}))
            stack.enter_context(patch.object(job, 'verify', return_value=dict(graceful_stop_seconds=85200)))
            stack.enter_context(patch.object(job, 'check_predecessor', return_value=(previous, {})))
            stack.enter_context(patch.object(job, 'predecessor_complete', side_effect=completed))
            for key in ('submit', 'cancel_successor', 'verify', 'command_for'):
                stack.enter_context(patch.object(job.controller, key))
            summary = stack.enter_context(patch.object(job.controller, 'summarize'))
            phases = []
            def run(phase):
                phases.append(phase.name)
                if phase == previous:
                    with patch.object(job, 'BASE_COMMAND', return_value=['python', '-m', 'old']):
                        command = job.controller.command_for(phase, {}, {}, Path('/tmp/base'))
                    self.assertEqual(command[0], 'env')
                    self.assertIn(str(previous/'source'), command[1])
                    self.assertEqual(command[-1], 'scripts.cross_dataset_campaign.foundation_msd')
            stack.enter_context(patch.object(job.controller, 'run', side_effect=run))
            job.run(stage)
            return phases, summary.call_args

    def test_finishes_predecessor_before_new_runs(self):
        phases, _ = self.run_chain([False, True])
        self.assertEqual(phases, ['previous', 'new'])

    def test_archive_wait_does_not_start_new_runs(self):
        phases, summary = self.run_chain([False, False])
        self.assertEqual(phases, ['previous'])
        self.assertEqual(summary.args[2], 'waiting_for_archive_offline')

    def test_completed_predecessor_is_not_retrained(self):
        phases, _ = self.run_chain([True])
        self.assertEqual(phases, ['new'])


def support_check(stage):
    import torch
    from scripts.cross_dataset_campaign.foundation import Volumes, SupportSlabs
    torch.set_num_threads(2)
    job.verify(stage)
    p = read(stage/'protocols/GeoFWI3D/protocol.json')
    schedules = []
    results = []
    for arm in job.ARMS:
        protocol = {**p, 'arm': arm}
        support = SupportSlabs(Volumes(protocol), protocol, None)
        schedule = support.schedule(protocol)
        calibration = support.calibration_schedule(protocol['calibration_slabs_per_volume'])
        if len(support.inventory) != 2 or len(schedule) != 400 or len(calibration) != 8:
            raise ValueError('Wrong support, update sampling or calibration inventory')
        if schedule != support.schedule(protocol) or {case for case, _ in schedule} != set(p['train_cases']):
            raise ValueError('Training schedule is not deterministic or differs from selected supports')
        schedules.append(schedule)
        results.append(dict(arm=arm, inventory=support.inventory, training_pairs=len(schedule),
                            calibration_pairs=len(calibration), calibration_schedule=calibration))
    if any(s != schedules[0] for s in schedules):
        raise ValueError('Arms do not use matched training samples')
    write(stage/'support_preflight.json', dict(status='passed', validation_volumes=96, arms=results))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', type=Path); args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(HandoffTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(): raise SystemExit(1)
    if args.stage: support_check(args.stage)


if __name__ == '__main__':
    main()
