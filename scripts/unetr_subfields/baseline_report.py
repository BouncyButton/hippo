"""Maintain the native-baseline comparison from verified durable archives."""
import argparse
from pathlib import Path
import time
from scripts.cross_dataset_campaign.common import read,write
from scripts.cross_dataset_campaign.report_seed_baselines import report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    last=None
    while True:
        if not (args.report/'frozen/manifest.json').exists():
            if not args.watch:raise ValueError('Source archive is not ready')
            time.sleep(5);continue
        result=report(args.report)
        state=read(args.report/'live.json').get('status.json',{}) if (args.report/'live.json').exists() else {}
        result.update(live_status=state.get('status','starting'),active=state.get('active'),last_observed_quota=state.get('quota'))
        write(args.report/'comparison.json',result)
        marker=(result['completed'],result['live_status'],result['active'])
        if marker!=last:print(dict(completed=result['completed'],total=result['total'],status=result['live_status'],active=result['active']),flush=True);last=marker
        if not args.watch or result['live_status'] in ('complete','failed','paused'):return
        time.sleep(30)

if __name__=='__main__':main()
