"""Fixed-budget, case-balanced sampling for the MNI MedSAM3 baseline."""
import hashlib


DESCRIPTION = ('Twenty shuffled passes over forty crops. Each crop contributes twenty distinct '
               'nonoverlapping adjacent plane pairs from its original 48 planes.')


def schedule(protocol):
    cases = sorted(protocol['train_cases'])
    if (protocol['dataset'], protocol['updates'], protocol['accumulation'], len(set(cases)),
            len(cases)) != ('MNI', 400, 2, 40, 40):
        raise ValueError('This sampling experiment uses 40 MNI crops and 800 draws')

    def rank(*parts):
        value = '|'.join(map(str, (protocol['seed'], *parts)))
        return hashlib.sha256(value.encode('utf-8')).digest()

    anchors = {case: sorted(range(0, 48, 2), key=lambda a: rank('pair', case, a))[:20]
               for case in cases}
    order = []
    for epoch in range(20):
        for case in sorted(cases, key=lambda case: rank('order', epoch, case)):
            order.append([case, anchors[case][epoch]])
    return order


def install(slabs):
    def fixed_schedule(self, protocol):
        if set(self.indices) != set(protocol['train_cases']) or any(
                len(indices) != 48 for indices in self.indices.values()):
            raise ValueError('The actual MNI training cases or slice counts differ')
        return schedule(protocol)
    slabs.schedule = fixed_schedule
