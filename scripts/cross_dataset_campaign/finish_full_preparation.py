"""Finish local draft assembly when the running CPU profiles become available."""
from pathlib import Path
import subprocess
import sys
import time

from scripts.cross_dataset_campaign.common import MODELS, read, write
from scripts.cross_dataset_campaign.prepare_full import PROFILES, ROOT


def main():
    began=time.time()
    status=PROFILES.parent/'assembly_status.json'
    while True:
        missing=[model for model in MODELS if not (PROFILES/f'geofwi3d_{model}.json').exists()]
        failed=[model for model in missing if (PROFILES/f'geofwi3d_{model}.log').exists()
                and 'Traceback (most recent call last)' in (PROFILES/f'geofwi3d_{model}.log').read_text()]
        if failed or time.time()-began>6*3600:
            write(status,dict(status='needs_attention',missing_profiles=missing,failed_profiles=failed,
                submission_enabled=False,updated_at=time.time()))
            return
        if not missing:
            subprocess.run([sys.executable,'-m','scripts.cross_dataset_campaign.prepare_full'],cwd=ROOT,check=True)
            result=read(PROFILES.parent/'preparation.json')
            if result['missing_profiles']:
                raise ValueError('Assembly still lacks profiles')
            write(status,dict(status='draft_assembled_pending_exploration',submission_enabled=False,
                cells=result['cells'],manifest_sha256=result['manifest_sha256'],updated_at=time.time()))
            return
        write(status,dict(status='waiting_for_local_cpu_profiles',missing_profiles=missing,
            submission_enabled=False,updated_at=time.time()))
        time.sleep(30)


if __name__=='__main__':main()
