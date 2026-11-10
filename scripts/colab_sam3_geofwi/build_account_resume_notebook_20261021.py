"""Build the second-account notebook around the original, hash-bound bundle."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / 'reports/sam3_geofwi_f013_s012_126_20261019'
OUT = EXPORT / 'account_migration_20261021'
SCRIPTS = Path(__file__).resolve().parent
ACCOUNT = 'filippofocaccia2@gmail.com'
RESULTS = '1ABU0S3cjFNozoYH4tD1DR_5aVp0z-kBz'


def main():
    handoff = json.loads((OUT / 'handoff.json').read_text())
    if handoff.get('clean_stop_confirmed') is not True:
        raise ValueError('Original session has not confirmed a clean stop')
    state = json.loads((OUT / 'source_campaign_state.json').read_text())
    done = len(state['completed'])
    notebook = json.loads((EXPORT / 'SAM3_GeoFWI3D_126_folds013_seeds012.ipynb').read_text())
    notebook['cells'][0]['source'] = [
        '# SAM3: resume on filippofocaccia2@gmail.com\n\n'
        f'The previous session stopped cleanly with **{done}/126 complete**. '
        f'This notebook resumes the remaining **{126-done}**, including the interrupted evaluation.\n\n'
        '1. Sign into Colab and authorize Drive as **filippofocaccia2@gmail.com**.\n'
        '2. Select **A100 GPU / High RAM**, matching the original NVIDIA A100-SXM4-80GB environment.\n'
        '3. Choose **Runtime → Run all**. Keep the original notebook stopped.\n\n'
        'The first cell verifies and imports the saved campaign into this account’s results folder. '
        'Repeated execution preserves all later progress and skips completed runs. '
        'Data, model and recipe hashes, folds 0/1/3, seeds 0/1/2, 14 recipes, 5 volumes and 25 updates remain unchanged. '
        'The original data and archives are read-only inputs. Training and evaluation use local SSD; '
        'Drive API requests are paced, retried on transient errors and checked against server checksums. '
        'The notebook requires 8 GiB of free Drive space and does not mount Drive.\n\n'
        '**To stop:** interrupt the training cell once and wait for '
        '`Clean stop: server-verified recovery saved.` before disconnecting.\n'
    ]
    bootstrap = ''.join(notebook['cells'][1]['source'])
    bootstrap = bootstrap.replace('from replication_store import Cloud',
        "import replication_store\nreplication_store.ACCOUNT = " + repr(ACCOUNT) +
        "\nreplication_store.RESULTS = " + repr(RESULTS) + '\nfrom replication_store import Cloud')
    bootstrap = bootstrap.replace("cloud=Cloud('1EDy575RclOhGBxhuylaHszLA67wT80qU')",
                                  'cloud=Cloud(' + repr(RESULTS) + ')')
    helper = (SCRIPTS / 'migrate_account_20261021.py').read_text()
    launcher = (SCRIPTS / 'resume_other_account_20261021.py').read_text()
    transfer = (
        '# Verify access before staging the campaign or downloading bulky inputs.\n'
        'for asset in ASSETS + [WEIGHTS]: cloud.metadata(asset["id"])\n'
        '(BOOTSTRAP/"migrate_account_20261021.py").write_text(' + repr(helper) + ')\n'
        'from migrate_account_20261021 import migrate\n'
        'migrated_state=migrate(cloud, Path("/content/sam3_account_migration"), '
        + repr(handoff['source_sha256']) + ')\n'
        'LAUNCHER=Path("/content/sam3_resume_other_account.py")\n'
        'LAUNCHER.write_text(' + repr(launcher) + ')\n'
        'LAUNCHER.with_suffix(".json").write_text(json.dumps(' + repr(handoff) + '))\n'
    )
    bootstrap = bootstrap.replace('for asset in ASSETS:\n', transfer+'for asset in ASSETS:\n')
    notebook['cells'][1]['source'] = [bootstrap]
    run = ''.join(notebook['cells'][5]['source'])
    run = run.replace('SESSION_HOURS=11.0', 'SESSION_HOURS=22.0')
    run = run.replace("str(BUNDLE/'replication.py')", 'str(LAUNCHER)')
    notebook['cells'][5]['source'] = [run]
    notebook['cells'][6]['source'] = [''.join(notebook['cells'][6]['source']).replace(
        '1EDy575RclOhGBxhuylaHszLA67wT80qU', RESULTS)]
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
            ast.parse(''.join(cell['source']))
    name = 'SAM3_126_resume_filippofocaccia2.ipynb'
    notebook['metadata']['colab']['name'] = name
    notebook['metadata']['colab']['gpuType'] = 'A100'
    notebook['metadata']['colab']['machine_shape'] = 'hm'
    target = OUT / name
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False)+'\n')
    print(target)


if __name__ == '__main__':
    main()
