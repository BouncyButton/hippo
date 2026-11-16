"""Include diagnostics in immutable recovery; optimizer contains authoritative gate."""
import os
from pathlib import Path
import shutil
import tempfile
import time
from scripts.colab_hfh.medsam_recovery import RecoveryWriter as ReferenceWriter


class RecoveryWriter(ReferenceWriter):
    def capture(self, force=False):
        if self.future:
            if force or self.future.done():
                self.future.result()
            else:
                return
        if not force and time.monotonic()-self.last < self.interval:
            return
        checkpoint = self.output/'checkpoint_latest.pt'
        if not checkpoint.exists():
            return
        snapshot = Path(tempfile.mkdtemp(prefix='snapshot-',dir=self.output.parent))
        os.link(checkpoint,snapshot/checkpoint.name)
        for path in self.output.glob('*.json'):
            if path.name not in ('snapshot_manifest.json','latest.json'):
                shutil.copyfile(path,snapshot/path.name)
        self.future = self.pool.submit(self.persist,snapshot)
        self.last = time.monotonic()
