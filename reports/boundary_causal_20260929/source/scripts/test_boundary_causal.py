import numpy as np
import torch
from train_boundary_causal import spatial_metrics
from thesis.new_constraints.separated_edge import rule_losses
from thesis.new_constraints.training_augmentation import warp_pair


def probabilities(pred):
    return np.eye(3, dtype=np.float32)[pred].transpose(3,0,1,2)


def test_unique_voxel_and_face_counts():
    y = np.zeros((20,20,20), dtype=np.uint8)
    y[5:15,5:15,5:10] = 1
    y[5:15,5:15,10:15] = 2
    perfect = spatial_metrics(probabilities(y), y, y)
    assert perfect['all_wrong'] == 0 and perfect['cross_correct_transition'] == 1
    pred = y.copy()
    pred[5,8,8] = 0
    pred[4,9,9] = 1
    pred[8,8,8] = 2
    m = spatial_metrics(probabilities(pred),pred,y)
    assert m['all_wrong']==3 and m['ap_swaps']==1 and m['boundary_union_errors']==2
    assert m['inner_layer1_fn']==m['outer_layer1_fp']==1
    assert m['cross_correct_transition'] < 1
    assert m['all_wrong'] == m['all_fp']+m['all_fn']+m['ap_swaps']


def test_constraints_remain_valid_after_geometric_augmentation():
    y = torch.zeros((1,1,32,32,32), dtype=torch.long)
    y[:,:,9:23,9:23,9:16]=1
    y[:,:,9:23,9:23,16:23]=2
    _, warped, applied = warp_pair(y.float(),y,angles=[.15,.07,-.1],scale=1.05,translation_xyz=[1.,0.,-1.])
    assert applied
    z=torch.nn.functional.one_hot(warped[:,0],3).permute(0,4,1,2,3).float()*40-20
    terms,counts=rule_losses(z,warped)
    assert all(float(n)>0 for n in counts.values())
    assert all(float(v)<1e-10 for v in terms.values())
