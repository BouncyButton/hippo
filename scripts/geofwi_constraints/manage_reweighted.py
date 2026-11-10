"""Transfer and submit the reconciled full constraint suite without a collector."""
import argparse
from scripts.geofwi_constraints import prepare_reweighted as prepared
from scripts.geofwi_constraints import transfer, launch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('transfer','launch'));args=ap.parse_args()
    transfer.REMOTE_TIMEOUT=900  # Full-suite evidence includes thousands of small files.
    module=transfer if args.action=='transfer' else launch
    for key in ('ROOT','NAME','REPORT','REMOTE'):setattr(module,key,getattr(prepared,key))
    if args.action=='launch':
        assert prepared.read(prepared.REPORT/'superseded_cleanup.json')['status']=='complete'
    module.main()


if __name__=='__main__':main()
