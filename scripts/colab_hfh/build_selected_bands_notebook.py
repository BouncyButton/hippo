"""Persist selected recipes and resumable search in the existing notebook."""
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'reports/hfh_colab_exploration_20261111'
    source=root/'scripts/colab_hfh'
    selected=json.loads((out/'bands_analysis/selected_recipes/winning_recipes.json').read_text())['models']
    assert len(selected)==4
    files={'bands_recipe_search.py':source/'bands_search.py','bands_campaign_search.py':source/'bands_campaign.py',
           'bands_v1_payload.json':out/'bands_v1_payload.json'}
    for name in ('bands_worker.py','bands_archive.py'):
        files['bundle/source/scripts/colab_hfh/'+name]=source/name
    package=out/'hfh_band_search_assets.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,path in files.items():
            info=zipfile.ZipInfo(name,date_time=(2026,11,11,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,path.read_bytes())
    sha=hashlib.sha256(package.read_bytes()).hexdigest()
    path=out/'HFH_baseline_shared_seed_search.ipynb'
    notebook=json.loads(path.read_text())
    assert 'BAND_SEARCH_ASSET_SHA256' in notebook['cells'][4]['source']
    notebook['cells'][4]['source']=re.sub(r"BAND_SEARCH_ASSET_SHA256 = '[a-f0-9]+'",f"BAND_SEARCH_ASSET_SHA256 = '{sha}'",notebook['cells'][4]['source'])
    names={'nnunet':'nnU-Net','unetr_vit_b16':'UNETR','basic_unet16':'BasicUNet','swinunetr':'SwinUNETR'}
    text='''# HFH — selected BAND recipes, three seeds × three folds

**Completed:** all four models have a selected recipe with a positive mean Dice change on every fold and final Dice ≥50% in all nine seed/fold pairs. **34/36 individual pairs improved.** nnU-Net improves only slightly and has two small regressions; its gain is not evidence of statistical significance.

Compute: the configured compute account. Inputs, results and archives: the configured storage account. These comparisons ran on **L4 BF16**; an A100 is not required. Preserve matched precision. FP32 requires separate matched controls.

| Model | Baseline Dice | Selected BAND Dice | Change (pp) | Improved pairs |
|---|---:|---:|---:|---:|
'''
    for row in selected:
        text+=f"| {names[row['model']]} | {row['baseline_dice']:.3f}% | {row['selected_dice']:.3f}% | {row['delta_pp']:+.3f} | {row['improved_pairs']}/9 |\n"
    text+='''
| Model | Initial LR | Total updates | Native-only prefix | BAND ramp | Calibration target / p95 cap |
|---|---:|---:|---:|---:|---:|
'''
    for row in selected:
        text+=f"| {names[row['model']]} | {row['initial_lr']:g} | {row['updates']} | {row['native_prefix']} | {row['ramp']} | {row['calibration_target']:g} / {row['calibration_p95_cap']:g} |\n"
    text+='''
The native optimizer, learning-rate horizon, loss, batch size, initialization, data order, no augmentation and final-budget checkpoint criterion remain matched to the controls. Each fold uses five training subjects (ten hippocampal crops) and five validation subjects (ten crops). The optimization cap stays **31,976,450,667,090 FLOPs/run**; training-only calibration is separately accounted for. The reported Dice is mean per-crop foreground Dice on the localized grid, with synthetic padding discarded.

BAND remains balanced inner/outer BCE on **two six-connected morphological voxel steps**. Calibration uses only the training crops, after the stated native-only prefix. Ramp length is in updates; the calibrated weight is determined independently from training data for each run.

The original 36 BAND runs and all 40 additional recipe-search runs are retained. UNETR and SwinUNETR keep the original BAND recipe; nnU-Net uses `nnunet_r12`, BasicUNet uses `basic_unet16_r09`. The search index is `results_band_recipe_search`; candidate archives are under `results_band_search_*`, and original runs remain under `results_bands_v1`.

**These are validation-selected exploratory recipes.** Every attempt and individual regression is retained. Selection does not establish unbiased test-set improvements.

Cell 4 resumes the saved search. On a fresh runtime it restores verified archives and the completed search index rather than retraining selected runs. Cell 5 shows all controls, the original BAND comparison, selected recipes, and every paired delta. This is the same four-model notebook; no replacement notebook was created.
'''
    notebook['cells'][0]['source']=text
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    path.write_text(json.dumps(notebook,indent=1)+'\n')
    print('Updated existing notebook; asset SHA256',sha)


if __name__=='__main__':main()
