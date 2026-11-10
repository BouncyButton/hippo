"""Archive each finished recovery run and retain the live seed decisions."""
import argparse
import json
from pathlib import Path
from scripts.cross_dataset_campaign import collect
from scripts.cross_dataset_campaign.common import read, write

SNAPSHOT=collect.snapshot


def snapshot(stage):
    state=SNAPSHOT(stage)
    code='import sys;from pathlib import Path;p=Path(sys.argv[1]);print(p.read_text() if p.exists() else "{}")'
    state['selection.json']=json.loads(collect.remote(['python3','-c',code,stage/'selection.json']))
    # Exhaustion is a finite terminal state, not permission to expand the search.
    if state.get('status.json',{}).get('status')=='candidate_block_exhausted':
        state['status.json']['original_status']='candidate_block_exhausted'
        state['status.json']['status']='paused'
    return state


def main():
    parser=argparse.ArgumentParser()
    for name in ('stage','local-stage','destination'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();collect.snapshot=snapshot
    collect.serve(args.stage,args.local_stage,args.destination)
    write(args.destination/'selection.json',read(args.destination/'live.json').get('selection.json',{}))


if __name__=='__main__':main()
