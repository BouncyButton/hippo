"""Run the isolated ramp revision with unchanged sequential recovery transport."""
def campaign_source():
    from scripts.colab_hfh.residual_band_campaign import campaign_source as old
    source=old()
    changes=[("'scripts.colab_hfh.residual_band_worker'","'scripts.colab_hfh.residual_band_v2_worker'"),
        ('from scripts.colab_hfh.residual_band_archive import archive','from scripts.colab_hfh.residual_band_v2_archive import archive'),
        ("'residual_band.py','residual_band_worker.py'","'residual_band_v2.py','residual_band_v2_worker.py','residual_band_v2_protocol.py','residual_band_v2_archive.py','residual_band_v2_pilot.py','residual_band_campaign.py','residual_band.py','residual_band_worker.py'"),
        ("zipped.write(__file__,'source/residual_band_campaign.py')","zipped.write(__file__,'source/residual_band_v2_campaign.py')")]
    for a,b in changes:
        if source.count(a)!=1:raise ValueError('V2 campaign anchor changed')
        source=source.replace(a,b)
    return source


if __name__=='__main__':exec(compile(campaign_source(),__file__,'exec'),{'__name__':'__main__','__file__':__file__})
