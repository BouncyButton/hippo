"""Run a full-data scratch cell using the frozen trainer and native evaluator."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import time
from functools import partial

from scripts.cross_dataset_campaign import conventional as exp
from scripts.cross_dataset_campaign.common import digest, read, write
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("key")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda","cpu"))
    parser.add_argument("--smoke-steps", type=int, default=0)
    args = parser.parse_args()
    row = read(args.bundle / "cells.json")[args.key]
    args.protocol_dir = args.bundle / "protocols" / args.key
    p = read(args.protocol_dir / "protocol.json")
    args.fold, args.seed = row["partition"], row["seed"]
    args.model, args.objective, args.threads, args.resume = p["trainer_alias"], "dice", 4, True
    args.pkl = args.bundle / p["data_file"]
    args.output = args.work / args.key
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if p["augmentation"] != "mild_v1" or p["initial_lrs"][args.model] != row["lr"]:
        raise ValueError("Scratch recipe differs from the campaign manifest")
    ds = row["dataset"]
    if ds != "MNI":
        exp.configure(p, "dice")
    exp.base.build_loss_and_optimizer = exp.ORIGINAL_LOSS_OPTIMIZER
    if ds == "HFH":
        from scripts.colab_hfh.data import PreparedPatients
        from scripts.colab_hfh.baseline import evaluate
        os.environ.pop("HIPPO_HFH_GPU_CACHE", None)
        exp.base.PreparedPatients = PreparedPatients
        exp.base.evaluate = evaluate
        exp.base._load_pkl_dataframe = lambda path: {**read(path), "data": str(args.bundle / "data")}
    if ds == "MNI":
        from scripts.unetr_subfields import baseline_runtime, core
        spec = core.DATASETS["MNI"]
        exp.base.PreparedPatients = partial(core.PreparedSubfields, spec=spec)
        exp.base.evaluate = partial(core.evaluate, classes=spec["classes"])
        if row["model"] == "unetr_vit_b16":
            exp.base.build_model = lambda alias, arch: core.build_model(len(spec["classes"]), spec["spatial_size"])
        else:
            exp.base.build_model = lambda alias, arch: baseline_runtime.build_model(row["model"], arch)
        from scripts.capped_depth_delay.objectives import training_loss
        exp.trainer.training_loss = training_loss
        cost=p["models"][args.model]["objectives"]["dice"]["flops_per_update"]
        exp.base.learning_rate=lambda name,step,maximum,initial: exp.ORIGINAL_LR(name,step,p["lr_horizon_updates"],initial)
        exp.base.validation_steps=lambda maximum,count=20: sorted({min(maximum,max(1,t//cost)) for t in p["validation_target_flops"]}|{maximum})
    started = time.monotonic()
    write(args.output / "runtime_recipe.json", dict(dataset=ds, model=row["model"],
          host=socket.gethostname(), gpu=torch.cuda.get_device_name() if args.device=="cuda" else None,
          precision=p["precision"], initialization="from_scratch", loss=p["training_loss"],
          augmentation=p["augmentation"],initial_lr=row["lr"], optimizer="SGD/Nesterov" if args.model=="nnunet" else "AdamW",
          schedule_horizon=p["lr_horizon_updates"], archive_recipe_differences=p["archive_recipe_differences"],
          source_sha256=digest(Path(__file__)), primary_checkpoint="final_budget"))
    exp.trainer.run(args)
    summary = read(args.output / "completion.json")
    if summary["discarded_attempts"]:
        raise ValueError("Interrupted updates require recovery audit")
    trace = [json.loads(line) for line in (args.output/"exposure.jsonl").read_text().splitlines()]
    if [r["update"] for r in trace] != list(range(1,summary["updates"]+1)):
        raise ValueError("Observed exposure trace differs from successful updates")
    write(args.output / "coverage.json", dict(successful_updates=summary["updates"], eligible=row["eligible_crops"],
          sampled_crops=sorted({r["case"] for r in trace}), elapsed_seconds=time.monotonic()-started,
          peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=="cuda" else 0,
          peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=="cuda" else 0))


if __name__ == "__main__":
    main()
