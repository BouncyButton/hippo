from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts.cross_dataset_campaign.common import read,write,digest
from scripts.mni_medsam3_cluster.adaptive_policy import choose_seeds,next_trials,needs_checkpoint
from scripts.mni_medsam3_cluster.adaptive_worker import migrate
from scripts.mni_medsam3_cluster import adaptive_campaign as campaign


def plan():
    return dict(seeds=[0,1,2],threshold=.60,maximum_updates=1000,
        tasks=[dict(id=f'f{f}s{s}u{u}',fold=f,seed=s,updates=u) for f in range(3) for s in range(3) for u in range(400,1001,100)])


def row(seed,updates,dice,fold=0,time=1):
    return dict(id=f'f{fold}s{seed}u{updates}',fold=fold,seed=seed,updates=updates,validation_dice=dice,observed_at=time)


class AdaptiveTests(unittest.TestCase):
    def test_baselines_first_and_only_failed_seed_gets100_more(self):
        p=plan(); rows=[row(0,400,.4)]
        self.assertEqual([(x['seed'],x['updates']) for x in next_trials(p,rows,[],set())],[(1,400),(2,400),(0,500)])
        self.assertEqual([x['seed'] for x in next_trials(p,rows,[],{'f1s0u400','f0s1u400'})],[2,0])
        rows.append(row(1,400,.60))
        self.assertNotIn(1,[x['seed'] for x in next_trials(p,rows,[],set())])

    def test_first_two_qualifying_seeds_replicate_on_other_folds(self):
        rows=[row(2,400,.61,time=1),row(0,500,.62,time=2),row(1,600,.9,time=3)]
        selected=choose_seeds(rows)
        self.assertEqual(selected,[2,0])
        tasks=next_trials(plan(),rows,selected,set())
        self.assertEqual({(t['fold'],t['seed'],t['updates']) for t in tasks},{(1,0,400),(1,2,400),(2,0,400),(2,2,400)})

    def test_cap_and_checkpoint_dependency(self):
        p=plan(); rows=[row(s,1000,.5) for s in range(3)]
        self.assertEqual(next_trials(p,rows,[],set()),[])
        self.assertTrue(needs_checkpoint(row(0,400,.4),p,[],set()))
        self.assertFalse(needs_checkpoint(row(0,400,.4),p,[],{'f0s0u400'}))
        self.assertFalse(needs_checkpoint(row(0,400,.4),p,[1,2],set()))
        self.assertFalse(needs_checkpoint(row(0,1000,.4),p,[],set()))

    def test_third_seed_still_gets_its_initial400_trial(self):
        rows=[row(0,400,.65),row(1,400,.66)]
        tasks=next_trials(plan(),rows,[0,1],set())
        self.assertEqual((tasks[0]['fold'],tasks[0]['seed'],tasks[0]['updates']),(0,2,400))
        tasks=next_trials(plan(),rows+[row(2,400,.4)],[0,1],set())
        self.assertFalse(any(t['fold']==0 for t in tasks))

    def test_active_transfer_blocks_early_parent_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage=Path(tmp);p=plan();r=row(0,400,.4)
            with patch.object(campaign.base,'retire') as retire:
                campaign.retire_ready(stage,p,[r],[1,2],{'f0s0u500'})
                retire.assert_not_called()
                campaign.retire_ready(stage,p,[r],[1,2],set())
                retire.assert_called_once()

    def test_migration_keeps_optimizer_rng_adapter_history_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir();output=root/'output'
            old=dict(name='old',updates=400,seed=0,fold=0,train_cases=['a','b'],arm='dice')
            write(parent/'protocol.json',old);write(parent/'calibration.json',dict(weights={'bands':0.0,'edge':0.0}))
            state=dict(step=400,history=[dict(update=i+1,lr=5e-6) for i in range(400)],protocol_sha256=digest(parent/'protocol.json'),
                adapter={'a':torch.tensor([1.,2.])},optimizer={'state':{0:{'exp_avg':torch.tensor([.1,.2])}}},torch_rng=torch.get_rng_state())
            torch.save(state,parent/'checkpoint_latest.pt');original=digest(parent/'checkpoint_latest.pt')
            write(parent/'completion.json',dict(updates=400,checkpoint_sha256=original))
            target={**old,'name':'new','updates':500};target.pop('arm');write(root/'protocol.json',target)
            migrate(parent,output,root/'protocol.json')
            actual=torch.load(output/'checkpoint_latest.pt',weights_only=False)
            self.assertEqual(actual['step'],400);self.assertEqual(actual['history'],state['history'])
            torch.testing.assert_close(actual['adapter']['a'],state['adapter']['a'],rtol=0,atol=0)
            torch.testing.assert_close(actual['optimizer']['state'][0]['exp_avg'],state['optimizer']['state'][0]['exp_avg'],rtol=0,atol=0)
            self.assertTrue(torch.equal(actual['torch_rng'],state['torch_rng']))
            self.assertEqual(digest(parent/'checkpoint_latest.pt'),original)
            self.assertEqual(actual['protocol_sha256'],digest(output/'protocol.json'))
            self.assertEqual(read(output/'resume_provenance.json')['status'],'verified')
            target['seed']=1;write(root/'bad.json',target)
            with self.assertRaises(ValueError):migrate(parent,root/'bad_output',root/'bad.json')


if __name__=='__main__':unittest.main()
