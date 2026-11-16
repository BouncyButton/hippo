"""Runtime utilities for the archived HFH protocol."""
from scripts.colab_medsam3.runtime import read, write, digest, copy_verified

from scripts.colab_adni.runtime import restore_recovery as restore_snapshot

def publish_metadata(output, durable):
    for path in sorted(output.iterdir()):
        if path.is_file() and path.suffix in ('.json', '.csv', '.log', '.jsonl', '.png'):
            if path.name not in ('snapshot_manifest.json', 'supervisor_ready.json'):
                copy_verified(path, durable/path.name)
