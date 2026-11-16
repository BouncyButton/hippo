"""Request a controller pause only after one named run finishes evaluation."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run:
        raise ValueError('Expected a run basename')
    completion = args.work/'runs'/args.run/'completion.json'
    deadline = time.monotonic()+3600
    while time.monotonic() < deadline:
        if completion.exists():
            done = json.loads(completion.read_text())
            state = json.loads((args.work/'campaign_status.json').read_text())
            if state.get('active', {}).get('name') != args.run:
                print('MISSED_BOUNDARY: no pause requested', flush=True)
                return
            if done.get('status') != 'complete' or done.get('updates') != 200:
                raise ValueError('Incomplete run cannot trigger boundary pause')
            (args.work/'STOP').touch()
            record = dict(at=time.time(), run=args.run, updates=200,
                          reason='Install next bounded candidate bank after completed evaluation',
                          action='STOP requested; existing controller must verify/archive before exit')
            (args.work/'boundary_pause.json').write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(record), flush=True)
            return
        time.sleep(2)
    raise TimeoutError('Named run did not finish; no pause requested')


if __name__ == '__main__': main()
