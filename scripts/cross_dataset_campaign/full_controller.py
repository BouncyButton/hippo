"""Long-study worker using the tested archive/resume controller, release gated."""
from scripts.cross_dataset_campaign import controller
from scripts.cross_dataset_campaign.common import read, digest


def verify(stage):
    for name,sha in read(stage/'manifest.json').items():
        if digest(stage/name)!=sha:
            raise ValueError('Frozen full-study file changed: '+name)
    plan=read(stage/'plan.json')
    if not plan.get('submission_enabled') or not read(stage/'control.json').get('full_submission_enabled'):
        raise RuntimeError('Full study is gated until exploration review and explicit release')
    if plan['augmentation']!='none' or len(plan['tasks'])!=1863:
        raise ValueError('Wrong full-study matrix')
    if plan.get('required_preparation'):
        raise ValueError('Full-study profiles remain incomplete')
    for task in plan['tasks']:
        if not task.get('protocol'):
            raise ValueError('An exploration-dependent protocol has not been selected')
        p=read(stage/'protocols'/task['protocol']/'protocol.json')
        if p['augmentation']!='none':
            raise ValueError('Augmentation is forbidden')
        if task['kind']=='conventional' and p['profile_status']!='measured':
            raise ValueError('Constraint FLOPs have not been measured')
    return plan


def main():
    controller.verify=verify
    controller.main()


if __name__=='__main__':main()
