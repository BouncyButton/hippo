"""Mirror the conditional Swin recovery resolution with normal archive snapshots."""
import json

from scripts.cross_dataset_campaign import collect

ORIGINAL_SNAPSHOT=collect.snapshot


def snapshot(stage):
    state=ORIGINAL_SNAPSHOT(stage)
    code='import json,sys;from pathlib import Path;p=Path(sys.argv[1]);print(p.read_text() if p.exists() else "null")'
    resolution=json.loads(collect.remote(['python3','-c',code,stage/'recovery_resolution.json']))
    if resolution is not None:state['recovery_resolution.json']=resolution
    return state


if __name__=='__main__':
    collect.snapshot=snapshot
    collect.main()
