"""Use the verified search dispatcher with the explicitly shared budget."""
import argparse
from pathlib import Path
from scripts.colab_hfh import nnunet_search13_campaign as campaign
from scripts.colab_hfh.nnunet_search13_shared_policy import NAME,CAP,ARMS,candidates


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--max-workers',type=int,required=True)
    args = parser.parse_args()
    campaign.NAME,campaign.CAP,campaign.ARMS,campaign.candidates = NAME,CAP,ARMS,candidates
    campaign.run(args.root,args.package,args.max_workers)
