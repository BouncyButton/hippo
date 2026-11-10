"""Actual annotated A/P contacts, restricted to native foreground, in millimetres."""
import numpy as np
import torch
from scripts.band_chamfer.ap import ap_target
from scripts.edge_ablation.objectives import valid_field
from .geometry import Targets as OriginalTargets, move
from .ap82_matrix import VERSION


def build(labels, valid):
    t = ap_target(labels, valid, np.ones(3))
    pair = t['pair']
    groups = []
    for level in (0, 1):
        use = pair['scale'] == level
        n = int(use.sum())
        weight = torch.full((n,), 1/n if n else 0., dtype=torch.float32)
        groups.append(dict(inside=dict(index=pair['i'][use], weight=weight),
                           outside=dict(index=pair['j'][use], weight=weight.clone())))
    return dict(foreground=t['anterior'], valid=t['support'], distance=t['distance'], pairs=groups)


class Targets:
    def __init__(self, root, protocol, entry):
        self.entry, self.protocol = entry, protocol
        self.is_ap = entry.get('objective_version') == VERSION
        self.delegate = OriginalTargets(root, protocol, entry['base_objective'] if self.is_ap else entry)
        self.cached, self.stats = {}, {}

    def get(self, item, device):
        base = self.delegate.get(item, device)
        if not self.is_ap: return base
        name = item['case_name']
        if name not in self.cached:
            labels = item['label'][0].long().numpy()
            valid = valid_field(labels.shape, self.protocol['native_geometry'][name]['native_shape'])
            t = build(labels, valid)
            self.cached[name] = t
            width = self.entry['ap82']['width']
            support = t['valid'] & (t['distance'] <= width)
            self.stats[name] = dict(band_anterior=int((support & t['foreground']).sum()),
                band_posterior=int((support & ~t['foreground']).sum()),
                endpoint_pairs={str(d):len(g['inside']['index']) for d,g in zip((1,3),t['pairs'])},
                geometry='Actual label-1/label-2 contact; both endpoint paths stay in their class; native foreground only')
        return dict(base=base, ap82=move(self.cached[name], device))

    def statistics(self):
        if not self.is_ap: return self.delegate.statistics()
        return dict(base=self.delegate.statistics(), ap82=self.stats)
