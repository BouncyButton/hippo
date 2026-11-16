"""Run the fixed nine lower-budget native controls with existing recovery rules."""
from pathlib import Path


def campaign_source():
    source=Path(__file__).with_name('bands_campaign.py').read_text()
    replacements=[
        ('def extract_archive(path,durable):',
         'from scripts.colab_hfh.constraint_drive import Drive\n\ndef extract_archive(path,durable):'),
        ('from scripts.colab_hfh.bands_archive import archive',
         'from scripts.colab_hfh.lower_budget_protocol import archive_native as archive'),
        ("arm='dice_bands'","arm='dice'"),
        ("'scripts.colab_hfh.bands_worker'","'scripts.colab_hfh.baseline'"),
        ("for filename in ('bands_worker.py','bands_archive.py','baseline.py','runtime.py'):",
         "for filename in ('lower_budget_protocol.py','lower_budget_search.py','bands_campaign.py','constraint_drive.py','baseline.py','runtime.py'):"),
        ("zipped.write(__file__,'source/bands_campaign.py')",
         "zipped.write(__file__,'source/lower_budget_baseline_campaign.py')"),
        ("'source/bands_v1_payload.json'","'source/lower_budget_native_payload.json'")]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('Native budget campaign integration changed: '+old)
        source=source.replace(old,new)
    return source.replace('band_dice','lower_budget_dice').replace('BAND','LOWER_BASELINE').replace(' band=',' lower=')


if __name__=='__main__':
    exec(compile(campaign_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
