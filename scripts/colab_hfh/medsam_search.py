"""HFH MedSAM controller using the existing scheduler and compact archives."""
import os
from pathlib import Path


def controller_source():
    source = (Path(__file__).parent/'search.py').read_text()
    edits = [
        ('from scripts.colab_hfh.runtime import read, write, digest, copy_verified, archive, verify_archive',
         'from scripts.colab_hfh.runtime import read, write, digest, copy_verified\n'
         'from scripts.colab_hfh.medsam_archive import archive, verify_archive, has_archive, configure'),
        ("    write(lease/'owner.json', dict(pid=os.getpid(), started=time.time(), work=str(args.work)))\n    try:",
         "    write(lease/'owner.json', dict(pid=os.getpid(), started=time.time(), work=str(args.work)))\n    try:\n"
         "        if args.kind != 'medsam3': raise ValueError('MedSAM-only controller')\n"
         "        configure(args.bundle, args.work, args.durable)"),
        ("                if (durable/'archive_manifest.json').exists():", '                if has_archive(durable):'),
        ("                    if (durable/'cleanup.json').exists():", '                    if True:'),
        ("                output.mkdir(parents=True, exist_ok=True)",
         "                if (output/'completion.json').exists():\n"
         "                    result = archive(output, durable, args.kind, model, fold, seed, protocol)\n"
         "                    rows.append(result); scores[(model, fold)] = result['validation_dice']; continue\n"
         "                output.mkdir(parents=True, exist_ok=True)"),
        ("                copy_verified(output/'worker.log', durable/'worker.log')", '                # Worker log is included in the verified compact archive.'),
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError('HFH scheduler changed; review compact-controller integration')
        source = source.replace(before, after)
    return source


def main():
    os.environ['HFH_CONTROLLER_ARCHIVES'] = '1'
    namespace = {'__name__': '__main__', '__file__': __file__}
    exec(compile(controller_source(), __file__, 'exec'), namespace)


if __name__ == '__main__':
    main()
