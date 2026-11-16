"""Build the portable, immutable package for the two authorized tests."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from scripts.colab_hfh.nnunet_composition_calibration import derive_specification, make_protocol, ARMS, NAME
from scripts.colab_hfh.runtime import verify_archive

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'reports/hfh_colab_exploration_20261111'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    work = REPORT/NAME
    package = work/'package'
    package.mkdir(parents=True,exist_ok=True)
    references = {
        'native': REPORT/'nnunet_seed0_long_v1/native',
        'band_edge': REPORT/'nnunet_long_matrix_v1/dice_bands_edge',
        'band_edge_depth': REPORT/'nnunet_long_matrix_v1/dice_bands_edge_depth'}
    def read(path):return json.loads(path.read_text())
    for name,folder in references.items():
        verify_archive(folder/'results')
        for kind in ('results','protocol'):
            shutil.copytree(folder/kind,package/'controls'/name/kind,dirs_exist_ok=True)
    parent = read(references['band_edge']/'results/calibration.json')
    triple = read(references['band_edge_depth']/'results/calibration.json')
    designs = {}
    for variant,arm in ARMS.items():
        original = references['band_edge' if arm == 'dice_bands_edge' else 'band_edge_depth']/'protocol'
        target = package/'protocols'/variant
        shutil.copytree(original,target,dirs_exist_ok=True)
        spec = derive_specification(parent,triple,variant)
        p = make_protocol(read(original/'protocol.json'),spec)
        (target/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
        designs[variant] = dict(objective=arm,weights=spec['weights'],anchor_envelope=spec['expected_envelope'],
                               schedule=p['models']['nnunet']['objectives'][arm])
    for archived in (references['band_edge']/'source').iterdir():
        current = ROOT/'scripts/colab_hfh'/archived.name
        if sha(current) != sha(archived):
            raise ValueError('Shared scientific source drift: '+archived.name)
        target = package/'source/scripts/colab_hfh'/archived.name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(archived,target)
    for name in ('nnunet_composition_calibration.py','nnunet_composition_worker.py','nnunet_composition_campaign.py'):
        shutil.copyfile(ROOT/'scripts/colab_hfh'/name,package/'source/scripts/colab_hfh'/name)
    shutil.copyfile(REPORT/'unetr_completion_20261113/verification.json',package/'unetr_verification.json')
    shutil.copyfile(REPORT/'nnunet_research_decision_20261113/combination_findings.txt',package/'research_design.txt')
    shutil.copyfile(ROOT/'scripts/colab_hfh/launch_nnunet_composition.py',work/'launch_nnunet_composition.py')
    shutil.copyfile(ROOT/'scripts/colab_hfh/launch_nnunet_composition.py',package/'launch_nnunet_composition.py')
    shutil.copyfile(ROOT/'scripts/colab_hfh/test_nnunet_composition.py',package/'test_nnunet_composition.py')
    (package/'design.json').write_text(json.dumps(dict(status='prepared_not_started',arms=designs,
        native_validation_dice=read(references['native']/'results/result.json')['validation_dice'],
        execution_condition='Await explicit user go-ahead; then UNETR and MedSAM complete; idle L4; exact native prefix',
        source_policy='Separate execution bundle; no active shared source edits'),indent=2)+'\n')
    manifest = {str(p.relative_to(package)):sha(p) for p in sorted(package.rglob('*'))
                if p.is_file() and p.name != 'manifest.json'}
    (package/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive = work/(NAME+'_package.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zipped:
        for p in sorted(package.rglob('*')):
            if p.is_file():zipped.write(p,str(p.relative_to(package)))
    receipt = dict(package=str(archive),sha256=sha(archive),bytes=archive.stat().st_size,
                   files=len(manifest),variants=list(ARMS),status='prepared_not_started')
    (work/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':build()
