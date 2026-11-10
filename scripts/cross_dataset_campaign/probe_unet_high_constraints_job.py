"""Four matched 3D UNet Dice+CE constraint runs at 80% and 100%."""
from pathlib import Path
import shutil
import sys

from scripts.cross_dataset_campaign import probe_unet_constraints_job as pilot
from scripts.cross_dataset_campaign.common import digest, read, write

PROBE = 'msd_unet_ce_constraints_high_20261014'
REFERENCE = 'msd_unet_ce_constraints_20261014'


def configure():
    pilot.PROBE = PROBE
    pilot.TARGETS = (.8, 1.)
    pilot.P95_CAPS = {.8: 1., 1.: 1.25}
    pilot.AUTHORIZATION = (
        'User requests bands and bands+original edge+depth CE at 80% and 100%, '
        'all added to Dice+CE, using the same unaugmented MSD 3D UNet fold0 seed0 experiment. '
        'User delegates the p95 ceiling choice: preserve the 60% arm target-to-ceiling ratio '
        'of 1:1.25, giving ceilings 100% and 125%, fixed before observing outcomes.')
    pilot.JOB_NAME = 'hippo-unet-ce-high'
    pilot.JOB_MODULE = 'scripts.cross_dataset_campaign.probe_unet_high_constraints_job'


def main():
    configure()
    if len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        reference = pilot.ROOT/'reports'/REFERENCE
        destination = pilot.ROOT/'reports'/PROBE
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('profile.json', 'historical_selection.json'):
            shutil.copyfile(reference/name, destination/name)
        # Magnitudes are scalar multipliers on exactly the already-profiled graphs.
        old_source = reference/'frozen/source'
        graph_files = ('scripts/cross_dataset_campaign/probe_unet_constraints.py',
                       'scripts/cross_dataset_campaign/conventional.py',
                       'scripts/capped_depth_delay/objectives.py',
                       'baselines/simple_segmentation.py')
        hashes = {}
        for name in graph_files:
            current = digest(pilot.ROOT/name)
            if current != digest(old_source/name):
                raise ValueError('Scientific implementation changed since the six-run pilot: '+name)
            hashes[name] = current
        write(destination/'extension_provenance.json', dict(
            reference_campaign=REFERENCE,
            reference_manifest_sha256=digest(reference/'frozen/manifest.json'),
            reference_comparison_sha256=digest(reference/'comparison.json'),
            unchanged_graph_hashes=hashes,
            target_to_p95_ratio=1.25,
            rationale='Preserve the 60% arm margin, allowing actual 80% and 100% medians instead of clipping both to the old 75% ceiling.',
            profile_sha256=digest(destination/'profile.json'),
            profile_reuse='Only positive scalar coefficient magnitudes differ; tensor shapes and operation graphs are identical.'))
    pilot.main()


if __name__ == '__main__':
    main()
