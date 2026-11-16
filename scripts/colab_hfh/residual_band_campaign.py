"""Isolated transport for a conditional-BAND pilot; canonical matrix stays intact."""
def campaign_source():
    from scripts.colab_hfh.constraint_campaign import campaign_source as original
    source=original()
    changes=[
        ("'scripts.colab_hfh.constraint_worker'","'scripts.colab_hfh.residual_band_worker'"),
        ('from scripts.colab_hfh.constraint_archive import archive','from scripts.colab_hfh.residual_band_archive import archive'),
        ("'constraint_worker.py','constraint_archive.py'","'residual_band.py','residual_band_worker.py','residual_band_archive.py','residual_band_protocol.py','residual_band_pilot.py','constraint_worker.py','constraint_archive.py'"),
        ("zipped.write(__file__,'source/constraint_campaign.py')","zipped.write(__file__,'source/residual_band_campaign.py')"),
    ]
    for old,new in changes:
        if source.count(old)!=1:raise ValueError('Conditional campaign integration changed')
        source=source.replace(old,new)
    return source


if __name__=='__main__':
    exec(compile(campaign_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
