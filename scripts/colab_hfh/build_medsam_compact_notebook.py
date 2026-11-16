"""Embed the tested storage-only MedSAM overlay in the runnable notebook."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/hfh_colab_exploration_20261111'
TARGET = REPORT/'HFH_MedSAM3_200_3fold_3seed.ipynb'
MARKER = '# HFH compact MedSAM storage overlay'


def main():
    backup = REPORT/'before_compact_medsam'/TARGET.name
    backup.parent.mkdir(exist_ok=True)
    if not backup.exists():
        shutil.copyfile(TARGET, backup)
    notebook = json.loads(TARGET.read_text())
    sources = {name: (ROOT/'scripts/colab_hfh'/name).read_text() for name in
        ('medsam.py', 'medsam_recovery.py', 'medsam_archive.py', 'medsam_search.py')}
    setup = notebook['cells'][3]['source']
    if isinstance(setup, list): setup = ''.join(setup)
    setup = setup.split(MARKER)[0].rstrip()+'\n\n'
    setup += MARKER+'\nimport hashlib\n'
    setup += 'storage_sources = '+repr(sources)+'\n'
    setup += "for filename, source_text in storage_sources.items():\n    (BUNDLE/'source/scripts/colab_hfh'/filename).write_text(source_text)\n"
    setup += "storage_manifest = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in storage_sources.items()}\n"
    setup += "(BUNDLE/'medsam_storage_patch_manifest.json').write_text(json.dumps(storage_manifest, indent=2))\n"
    setup += "print('Compact MedSAM archives installed; 200-update scientific recipe unchanged.', flush=True)\n"
    notebook['cells'][3]['source'] = setup
    run = notebook['cells'][4]['source']
    if isinstance(run, list): run = ''.join(run)
    notebook['cells'][4]['source'] = run.replace("'scripts.colab_hfh.search'", "'scripts.colab_hfh.medsam_search'")
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []; cell['execution_count'] = None
    TARGET.write_text(json.dumps(notebook, indent=1)+'\n')
    print(TARGET)
    print(json.dumps({name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}, indent=2))


if __name__ == '__main__':
    main()
