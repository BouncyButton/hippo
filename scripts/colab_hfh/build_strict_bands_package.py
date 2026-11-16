"""Rebuild strict-search assets and update the existing notebook checksum."""
import hashlib
import json
from pathlib import Path
import re
import zipfile


def main():
    repository = Path(__file__).resolve().parents[2]
    source = repository / 'scripts/colab_hfh'
    export = repository / 'reports/hfh_colab_exploration_20261111'
    files = {
        'bands_strict_search.py': source / 'bands_strict_search.py',
        'bands_campaign_strict.py': source / 'bands_campaign.py',
        'bands_v1_payload.json': export / 'bands_v1_payload.json',
    }
    for name in ('bands_worker.py', 'bands_archive.py', 'bands_variants.py', 'bands_search.py'):
        files['bundle/source/scripts/colab_hfh/' + name] = source / name
    package = export / 'hfh_band_strict_assets.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            info = zipfile.ZipInfo(name, (2026, 11, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    digest = hashlib.sha256(package.read_bytes()).hexdigest()
    path = export / 'HFH_baseline_shared_seed_search.ipynb'
    notebook = json.loads(path.read_text())
    cell = notebook['cells'][4]
    text = ''.join(cell['source'])
    updated, count = re.subn(r"BAND_STRICT_ASSET_SHA256 = '[0-9a-f]+'",
                            f"BAND_STRICT_ASSET_SHA256 = '{digest}'", text)
    if count != 1:
        raise ValueError('Expected one strict package checksum in existing notebook')
    cell['source'] = updated
    path.write_text(json.dumps(notebook, indent=1))
    print(digest)


if __name__ == '__main__':
    main()
