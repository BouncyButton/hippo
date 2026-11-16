"""Run the full-coverage medical control with a separate name and verified archive."""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile


def covered_schedule(original, indices, protocol):
    """Keep the native sequence unless it misses cases. Replace only duplicate draws."""
    order = list(original)
    assert len(order) == protocol["updates"] * protocol["accumulation"]
    assert len(order) >= len(indices)
    count = Counter(case for case, _ in order)
    missing = sorted(set(indices) - set(count))
    rng = random.Random(protocol["seed"])
    rng.shuffle(missing)
    positions = list(range(len(order)))
    rng.shuffle(positions)
    for pos in positions:
        if not missing:
            break
        previous = order[pos][0]
        if count[previous] > 1:
            replacement = missing.pop()
            count[previous] -= 1
            order[pos] = (replacement, rng.randrange(len(indices[replacement])))
    assert {case for case, anchor in order} == set(indices)
    assert all(0 <= anchor < len(indices[case]) for case, anchor in order)
    return order


def worker(protocol, work, recovery):
    reservation = work.parent / 'external_workers.json'
    if reservation.is_file():
        key = protocol.stem.removesuffix('_coverage_control')
        if key in json.loads(reservation.read_text()):
            from continue_colab import existing, session
            drive = session()
            print('Waiting for the assigned external worker', key, flush=True)
            while (reservation.is_file() and key in json.loads(reservation.read_text())
                   and not existing(drive, protocol.stem + '_results.zip')):
                time.sleep(90)
            if existing(drive, protocol.stem + '_results.zip'):
                # A parent imported before assignment will reopen its queue and reuse the publication.
                raise SystemExit(75)
    from scripts.train_medsam3_constraint_few_shot import Slabs
    import medical_worker
    native = Slabs.schedule
    def schedule(self, config):
        return covered_schedule(native(self, config), self.indices, config)
    Slabs.schedule = schedule
    sys.argv = [sys.argv[0], str(protocol), str(work), str(recovery)]
    medical_worker.main()


def run(ds, last_hfh=False, only_key=None, offline=False):
    from continue_colab import existing, publish, session, validate_output
    bundle = Path("/content/full_data_" + ds)
    medical = Path(str(bundle) + "_medical")
    root = Path("/content/covered_medical_" + ds)
    for child in ("protocols", "work", "recovery"):
        (root / child).mkdir(parents=True, exist_ok=True)
    driver = Path(__file__).resolve()
    original_worker = medical / "medical_worker.py"
    if only_key:
        assert (ds == 'HFH' or offline) and not last_hfh
        assert any(p.stem == only_key for p in (medical / 'protocols').glob('*.json'))
    lock_name = f'{only_key}.lock' if only_key else ('secondary.lock' if last_hfh else 'queue.lock')
    guard = (root / lock_name).open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if offline and not only_key:
        raise ValueError('Direct archive collection requires one assigned run')
    drive = None if offline else session()
    # Let in-flight historical evaluations publish under their original names.
    marker = root / "supersedes_new_legacy_workers"
    for entry in (() if offline else (original_worker, bundle / "worker.py")):
        body = entry.read_text()
        label = "# corrected full-coverage queue"
        if label not in body:
            main = body.rfind('if __name__')
            assert main >= 0 and '__main__' in body[main:]
            replacement = (label + '\nif __name__ == "__main__" and __import__("pathlib").Path('
                           + repr(str(marker)) + ').exists():\n'
                           + '    raise SystemExit("Superseded by the full-coverage queue")\n\n')
            temporary = entry.with_suffix(".next")
            temporary.write_text(body[:main] + replacement + body[main:])
            temporary.replace(entry)
    marker.touch()
    while True:
        active = []
        for proc in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                argv = proc.read_bytes().split(b"\0")
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
            if os.fsencode(original_worker) in argv:
                active.append(proc.parent.name)
        if not active:
            break
        print("Waiting for original medical worker", active, flush=True)
        time.sleep(40)
    src = bundle / "source"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, [medical, bundle / "compatibility",
        src, src / "scripts", src / "baselines/nnUNet"])), PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", WANDB_MODE="disabled")
    sources = sorted((medical / "protocols").glob("*.json"))
    if only_key:
        sources = [p for p in sources if p.stem == only_key]
    elif (root / 'external_workers.json').is_file():
        reserved = set(json.loads((root / 'external_workers.json').read_text()))
        sources = [p for p in sources if p.stem not in reserved]
    if last_hfh:
        assert ds == "HFH"
        # Take one late fold only while the main worker is several runs away.
        available = int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
        if available < 11600:
            print("Waiting for GPU headroom:", available, flush=True)
            return
        remaining = [p for p in sources if not existing(drive,p.stem + "_coverage_control_results.zip")]
        if len(remaining) < 4:
            return
        sources = [remaining[-1]]
        if (root / "work" / (sources[0].stem + "_coverage_control") / "worker.lock").exists():
            return
    for source in sources:
        p = json.loads(source.read_text())
        key = p["name"]
        target = key + "_coverage_control"
        if drive is not None and existing(drive, target + "_results.zip"):
            continue
        # Reuse any publication that already satisfies the same scientific protocol.
        prior = existing(drive, key + "_results.zip") if drive is not None else None
        if prior and ds != "HFH":
            import io
            data = drive.files().get_media(fileId=prior["id"]).execute(num_retries=4)
            assert hashlib.sha256(data).hexdigest() == prior["sha256Checksum"]
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                config = json.loads(z.read("protocol.json"))
                schedule = json.loads(z.read("slab_schedule.json"))
                done = json.loads(z.read("completion.json"))
                assert config == dict(p, arm="dice") and done["updates"] == p["updates"]
                if {c for c,a in schedule} == set(p["train_cases"]):
                    print("REUSED", key, prior["id"], flush=True)
                    continue
        p.update(name=target, sampling="Native crop and anchor sequence with duplicate draws replaced when needed for full coverage.",
                 learning_rate_horizon_updates=200 if ds == "HFH" else p["learning_rate_horizon_updates"])
        if ds == "HFH":
            p["archive_recipe_differences"]["full_data_lr_horizon_updates"] = 200
        spec = root / "protocols" / (target + ".json")
        if spec.exists():
            assert json.loads(spec.read_text()) == p
        else:
            spec.write_text(json.dumps(p, indent=2))
        out = root / "work" / target
        out.mkdir(exist_ok=True)
        if not (out / "completion.json").exists():
            print("TRAIN", target, flush=True)
            with (out / "training.log").open("a") as log:
                subprocess.run([str(bundle / "venv/bin/python"), str(driver), "--worker",
                                str(spec), str(root / "work"), str(root / "recovery")],
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        dice, covered = validate_output(target, root, True)
        assert covered == len(p["train_cases"])
        receipt = dict(key=key, dice=dice, observed=covered, eligible=len(p["train_cases"]),
                       updates=p["updates"], lr_horizon=p["learning_rate_horizon_updates"])
        (out / "coverage_receipt.json").write_text(json.dumps(receipt))
        path = root / (target + "_results.zip")
        with zipfile.ZipFile(path.with_suffix(".partial"), "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.iterdir()):
                if f.is_file() and f.suffix in (".json", ".jsonl", ".log", ".csv"):
                    z.write(f, f.name)
            z.write(source, "protocols/original.json")
            z.write(spec, "protocols/input.json")
            z.write(original_worker, "medical_worker.py")
            z.write(driver, "covered_medical.py")
        path.with_suffix(".partial").replace(path)
        if offline:
            with zipfile.ZipFile(path) as archive:
                assert archive.testzip() is None and json.loads(archive.read('coverage_receipt.json')) == receipt
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            print(json.dumps(dict(receipt, direct_archive=str(path), sha256=checksum)), flush=True)
            continue
        proof = publish(drive, key, path)
        print(json.dumps(dict(receipt, id=proof["id"])), flush=True)
        for f in (out / "checkpoint_latest.pt", root / "recovery" / target / "checkpoint_latest.pt",
                  root / "recovery" / target / "recovery" / "checkpoint_latest.pt"):
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", choices=["HFH", "ADNI", "MSD"])
    parser.add_argument("--worker", nargs=3, type=Path)
    parser.add_argument("--last-hfh", action="store_true")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--only-key")
    selection.add_argument("--offline-key", help="Train one assigned run for direct archive download")
    args = parser.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        run(args.dataset, args.last_hfh, args.only_key or args.offline_key, bool(args.offline_key))
