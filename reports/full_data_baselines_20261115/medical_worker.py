"""Bind the released medical adapters to the original per-dataset data and evaluator."""
import argparse
import fcntl
import os
from pathlib import Path
import sys

from scripts.colab_mni_medsam3 import optimized
from scripts.cross_dataset_campaign import foundation
from scripts.cross_dataset_campaign.common import digest, read


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("protocol",type=Path)
    parser.add_argument("work",type=Path)
    parser.add_argument("recovery",type=Path)
    args=parser.parse_args()
    p=read(args.protocol)
    out=args.work/p["name"]
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/"worker.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert set(p["train_cases"]).isdisjoint(p["evaluation_cases"])
    if p["dataset"]=="HFH":
        from scripts.colab_hfh.data import Volumes
        optimized.Volumes=Volumes
        optimized.SupportSlabs=foundation.SupportSlabs
    elif p["dataset"]=="ADNI":
        assert digest(p["data"])==p["dataset_sha256"]
        optimized.Volumes=foundation.Volumes
        optimized.SupportSlabs=foundation.SupportSlabs
    elif p["dataset"]=="MSD":
        from scripts import train_medsam3_multiclass as original
        assert p["classes"]==original.CLASSES and {int(c):name for c,name in p["prompts"].items()}==original.PROMPTS
        for name,expected in p["data_hashes"].items():
            assert digest(Path(p["data"])/name)==expected
        def slabs(volumes,protocol,processor):
            return original.MulticlassSlabs(Path(protocol["data"]),protocol["train_cases"],processor)
        def forward(model,criterion,slabs,case,anchor):
            native,bands,edge=original.forward_slab(model,criterion,slabs,case,anchor)
            return dict(native=native,bands=bands,edge=edge)
        optimized.Volumes=foundation.Volumes
        optimized.SupportSlabs=slabs
        foundation.forward_slab=forward
        foundation.forward_plane=lambda model,processor,pixels,protocol,labels=None,criterion=None: original.forward_plane(
            model,processor,pixels,labels,criterion)
    else:
        raise ValueError("Unknown medical dataset")
    os.environ["MNI_MANIFEST_SHA"]=digest(args.protocol)
    sys.argv=[sys.argv[0],"--protocol",str(args.protocol),"--output",str(out),
              "--durable",str(args.recovery/p["name"]),"--base-weights","/content/full_baseline_medical_inputs/sam3.pt"]
    optimized.main()


if __name__=="__main__":
    main()
