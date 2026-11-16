"""Hand-constructed spatial errors and independent checks of diagnostic totals."""
import numpy as np
from scipy import ndimage as ndi
from run import FACE, counts

y=np.zeros((19,29,17),dtype='uint8')
y[6:13,4:25,6:11]=1
y[6:13,16:25,6:11]=2
affine=np.eye(4)
p=y.copy()
a=counts(y,p,affine,3)
assert a['broken']==a['mismatches']==a['class_errors']==0
assert a['dice']==1
p[6,10,8]=0  # one isolated outer contour miss
p[9,10,1]=1  # one distant false positive
p[8,16,8]=1  # swap on an internal class boundary
r=counts(y,p,affine,3)
assert r['mismatches']==3 and r['class_errors']==4 and r['union_errors']==2
assert r['isolated']==r['centers']==r['band_union_errors']==1
assert r['far']==r['far4']==1
mask=y>0
# Independently count eligible unordered faces and prediction disagreements, voxel by voxel.
shell=(ndi.distance_transform_cdt(mask,'taxicab')+ndi.distance_transform_cdt(~mask,'taxicab'))<=2
faces=broken=0
for u in zip(*np.where(shell)):
    for axis in range(3):
        v=list(u)
        v[axis]+=1
        if v[axis]>=mask.shape[axis]:
            continue
        v=tuple(v)
        if shell[v] and mask[u]==mask[v]:
            faces+=1
            broken+=(p[u]>0)!=(p[v]>0)
assert faces==r['faces'] and broken==r['broken']
print('Passed perfect predictions, singleton, far false positive, internal swap, exhaustive face enumeration.')
