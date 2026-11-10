"""Protect the pilot's compute ceiling and distinguish completion from passing."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.colab_geofwi_seed_search import pilot


class PilotTests(unittest.TestCase):
    def test_no_additional_task_allowed(self):
        for args in [(1,0,pilot.MODEL), (0,1,pilot.MODEL), (0,0,'nnunet')]:
            with self.assertRaises(ValueError):
                pilot.task(*args)

    def test_completed_below_threshold_is_not_baseline_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            pilot.io.write(path/'dice_control.json',dict(final_dice=.29))
            row=dict(**pilot.task(),final_dice=.5)
            result=pilot.publish(path,[row])
            self.assertEqual(result['status'],'complete')
            self.assertFalse(result['threshold_passed'])
            self.assertTrue((path/'PILOT_COMPLETE.json').exists())
            self.assertFalse((path/'COMPLETE.json').exists())

    def test_paused_is_never_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            pilot.io.write(path/'dice_control.json',dict(final_dice=.29))
            result=pilot.publish(path,[],'paused')
            self.assertEqual(result['status'],'paused')
            self.assertFalse((path/'PILOT_COMPLETE.json').exists())

    def test_engine_archives_one_run_then_stops(self):
        e=pilot.engine
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'bundle';root.mkdir()
            durable=Path(tmp)/'results';durable.mkdir()
            pilot.io.write(durable/'dice_control.json',dict(final_dice=.29))
            pilot.io.write(durable/'preflight.json',dict(status='passed',binding='test'))
            folder=root/'protocol';folder.mkdir()
            for name in ['protocol.json','splits_10patients.json']:
                pilot.io.write(folder/name,{})
            output=root/'work'/pilot.RUN_ID;output.mkdir(parents=True)
            pilot.io.write(output/'completion.json',{})
            row=dict(**pilot.task(),final_dice=.4,initialization_sha256=pilot.INITIALIZATION)
            with patch.multiple(e, publish=pilot.publish, decide=pilot.decide), \
                 patch.object(e,'verify'), patch.object(e.io,'binding',return_value={'binding':'test'}), \
                 patch.object(e,'load_rows',return_value=[]), \
                 patch.object(e,'protocol',return_value=(folder,{})), \
                 patch.object(e.io,'check_drive_headroom'), \
                 patch.object(e.shutil,'disk_usage',return_value=type('Disk',(),{'free':100*1024**3})()), \
                 patch.object(e,'validate_output',return_value=row), \
                 patch.object(e.io,'archive_completed') as archive, \
                 patch.object(e,'result',return_value=row), patch.object(e.subprocess,'Popen') as process:
                e.run(root,durable,1,1)
                archive.assert_called_once()
                process.assert_not_called()
                self.assertEqual(pilot.io.read(durable/'pilot_status.json')['status'],'complete')
                # A completed result is reused without another training/archival attempt.
                with patch.object(e,'load_rows',return_value=[row]):
                    e.run(root,durable,1,1)
                archive.assert_called_once()
                process.assert_not_called()


if __name__=='__main__':
    unittest.main()
