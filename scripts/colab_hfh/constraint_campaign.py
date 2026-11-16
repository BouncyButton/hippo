"""Reuse the verified sequential transport/recovery loop for one objective.

Screening may fail fast. Frozen full-matrix payloads have no rejection rule and
therefore retain all nine outcomes, including regressions.
"""
from pathlib import Path


def campaign_source():
    source = Path(__file__).with_name('bands_campaign.py').read_text()
    replacements = [
        ('def extract_archive(path,durable):',
         'from scripts.colab_hfh.constraint_drive import Drive\n\n\ndef extract_archive(path,durable):'),
        ('from scripts.colab_hfh.bands_archive import archive', 'from scripts.colab_hfh.constraint_archive import archive'),
        ("arm='dice_bands'", "arm=payload['objective']"),
        ("'scripts.colab_hfh.bands_worker'", "'scripts.colab_hfh.constraint_worker'"),
        ("for filename in ('bands_worker.py','bands_archive.py','baseline.py','runtime.py'):",
         "for filename in ('constraint_worker.py','constraint_archive.py','constraint_schedule.py','constraint_calibration.py','constraint_geometry.py','constraint_drive.py','constraint_diagnostics.py','parameter_gradient_diagnostics.py','constraint_online_sdf.py','bands_variants.py','bands_campaign.py','baseline.py','runtime.py'):"),
        ("zipped.write(__file__,'source/bands_campaign.py')", "zipped.write(__file__,'source/constraint_campaign.py')"),
        ("'source/bands_v1_payload.json'", "'source/constraint_payload.json'"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Sequential campaign integration changed: '+before)
        source = source.replace(before, after)
    source = source.replace('band_dice', 'constraint_dice').replace('BAND', 'CONSTRAINT')
    source = source.replace(' band=', ' constraint=')
    # Keep the shared GPU lock, so BAND and multi-constraint jobs cannot overlap.
    return source


if __name__ == '__main__':
    exec(compile(campaign_source(), __file__, 'exec'), {'__name__': '__main__', '__file__': __file__})
