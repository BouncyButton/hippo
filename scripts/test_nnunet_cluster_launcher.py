"""Offline launch-policy regression checks; never submits or trains."""
from pathlib import Path
import re
import subprocess
import unittest
import json
import tempfile
from verify_nnunet_cache import expected_cases


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.path=Path(__file__).with_name('run_nnunet_cluster.sh')
        self.text=self.path.read_text().replace('\\\n',' ')

    def test_shell_syntax(self):
        subprocess.run(['bash','-n',str(self.path)],check=True)

    def test_only_required_configuration_and_bounded_workers(self):
        command=next(l.strip() for l in self.text.splitlines() if l.strip().startswith('nnUNetv2_plan_and_preprocess '))
        for flag in ('-c 3d_fullres','-npfp 2','-np 2','--verbose','--verify_dataset_integrity'):
            self.assertIn(flag,command)

    def test_failures_and_threads_are_explicit(self):
        self.assertIn('set -Eeuo pipefail',self.text)
        self.assertRegex(self.text,r"trap .* ERR")
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','PYTHONFAULTHANDLER'):
            self.assertIn(name+'=1',self.text)
        self.assertIn('nnUNet_n_proc_DA=2',self.text)

    def test_scientific_target_is_unchanged(self):
        self.assertIn('TRAINER="nnUNetTrainer_50epochsEarlyStopping"',self.text)
        self.assertIn('len(split["train"]) != 208 or len(split["val"]) != 52',self.text)
        self.assertIn('3d_fullres "${FOLD}" -tr "${TRAINER}"',self.text)

    def test_cache_recovery_and_local_io_are_verified(self):
        self.assertIn('--preprocessed-source)',self.text)
        self.assertEqual(self.text.count('scripts/verify_nnunet_cache.py'),2)
        self.assertIn('export nnUNet_preprocessed="${LOCAL_CACHE}"',self.text)
        self.assertIn('nnUNet_def_n_proc=2',self.text)
        self.assertIn('#SBATCH --time=02:00:00',self.text)

    def test_cache_case_set_and_overlap_guards(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'labelsTr').mkdir()
            cases=[f'case_{i:03d}' for i in range(260)]
            for case in cases:(root/'labelsTr'/(case+'.nii.gz')).touch()
            path=root/'splits.json'
            split={'train':cases[:208],'val':cases[208:]}
            path.write_text(json.dumps([split]))
            self.assertEqual(expected_cases(root,path,0),set(cases))
            split['val'][0]=split['train'][0]
            path.write_text(json.dumps([split]))
            with self.assertRaises(AssertionError):expected_cases(root,path,0)
            split['val']=cases[208:]
            split['val'][0]='unknown_case'
            path.write_text(json.dumps([split]))
            with self.assertRaises(AssertionError):expected_cases(root,path,0)


if __name__=='__main__':unittest.main()
