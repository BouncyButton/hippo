"""Contracts that prevent lost resume state, accidental augmentation and unsafe cleanup."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.cross_dataset_campaign.common import check_no_augmentation, confusion_metrics, read, write, digest
from scripts.cross_dataset_campaign.storage import retire
from scripts.cross_dataset_campaign.prepare import schedule


class Contracts(unittest.TestCase):
    def test_augmentation_rejected(self):
        for value in ('mild_v1',None,'random'):
            with self.assertRaises(ValueError):check_no_augmentation(dict(augmentation=value))
        check_no_augmentation(dict(augmentation='none'))

    def test_confusion_separates_swaps(self):
        result=confusion_metrics([[10,2,3],[4,20,5],[6,7,30]],['background','anterior','posterior'])
        self.assertEqual(result['foreground_fp'],5)
        self.assertEqual(result['foreground_fn'],10)
        self.assertEqual(sum(result['ap_swaps'].values()),12)
        self.assertEqual(result['per_class']['anterior']['fn'],9)

    def test_archive_requires_matching_receipt_and_unchanged_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);write(root/'completion.json',dict(status='complete'))
            (root/'checkpoint.pt').write_bytes(b'checkpoint')
            manifest=dict(metadata={'completion.json':dict(bytes=(root/'completion.json').stat().st_size,sha256=digest(root/'completion.json'))},
                deletion_candidates=[dict(name='checkpoint.pt',bytes=10,sha256=digest(root/'checkpoint.pt'))])
            write(root/'archive_manifest.json',manifest)
            with self.assertRaises(FileNotFoundError):retire(root)
            write(root/'archive_acknowledgment.json',dict(status='verified',manifest_sha256=digest(root/'archive_manifest.json'),
                source_verified=True,durable_archive='/Users/test/project/reports/run'))
            (root/'checkpoint.pt').write_bytes(b'CORRUPTION')
            with self.assertRaises(ValueError):retire(root)
            self.assertTrue((root/'checkpoint.pt').exists())
            (root/'checkpoint.pt').write_bytes(b'checkpoint')
            self.assertEqual(retire(root)['status'],'complete')
            self.assertFalse((root/'checkpoint.pt').exists())
            self.assertEqual(retire(root)['status'],'complete')

    def test_budget_exhausts_without_overshoot(self):
        from scripts.cross_dataset_campaign.common import CAP
        costs={'dice':dict(update_flops=355287609598,calibration_flops=0)}
        result=schedule(costs,'dice',dict(prefix_updates=0,sdf_prefix_updates=0,constraint_warmup_updates=9),10)
        self.assertLessEqual(result['consumed_training_flops'],CAP)
        self.assertLess(result['unused_flops'],result['flops_per_update'])

    def test_seismic_budget_is_eight_nnunet_passes(self):
        from scripts.cross_dataset_campaign.common import SEISMIC_CAP
        costs={'dice':dict(update_flops=1198540306205,calibration_flops=0)}
        result=schedule(costs,'dice',dict(prefix_updates=0,sdf_prefix_updates=0,constraint_warmup_updates=1),19,SEISMIC_CAP)
        self.assertEqual(result['max_steps'],152)
        self.assertEqual(result['patient_pass_epochs'],8.)
        self.assertEqual(result['unused_flops'],0)


if __name__=='__main__':unittest.main()
