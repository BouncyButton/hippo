"""Finite archive collector with a final per-model target comparison."""
from scripts.cross_dataset_campaign.collect import serve
from scripts.cross_dataset_campaign.common import REMOTE,read,write
from scripts.mni_constraints.target_search import ROOT,NAME


def main():
    report=ROOT/'reports'/NAME
    serve(REMOTE/NAME,ROOT/'experiments'/NAME,report)
    state=read(report/'live.json').get('status.json',{})
    if 'target_search' in state:write(report/'target_comparison.json',state['target_search'])


if __name__=='__main__':main()
