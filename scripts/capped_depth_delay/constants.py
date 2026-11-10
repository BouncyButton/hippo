"""Dependency-free scope identifiers for preparation/submission/collection."""
from itertools import combinations

CAMPAIGN = 'capped_depth_delay_31976_20261006'
SDF_ARMS = ('dice_sdf', 'dice_bands_sdf', 'dice_bands_edge_sdf')
DEPTH_ARMS = tuple('dice' + ''.join('_' + name for name in ('bands', 'edge', 'sdf') if name in names) + '_depth'
                   for count in range(4) for names in combinations(('bands', 'edge', 'sdf'), count))
ARMS = SDF_ARMS + DEPTH_ARMS


def base_arm(arm):
    return arm.removesuffix('_depth')
