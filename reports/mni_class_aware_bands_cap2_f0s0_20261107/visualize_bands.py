"""Render actual frozen MNI training bands; no model predictions or training."""
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from scripts.unetr_subfields import core
from scripts.unetr_subfields.contracts import DATASETS
from scripts.cross_dataset_campaign.common import digest
from thesis.new_constraints.bands.outer_boundary import build_boundary_bands

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STAGE = ROOT/'experiments/mni_class_aware_bands_cap2_f0s0_20261107'
COLORS = {1:'#ffc145', 2:'#3498ff', 3:'#b976ff'}
INNER, OUTER = '#00e2cf', '#ff526e'


def main():
    torch.set_num_threads(4)
    split = json.loads((STAGE/'protocols/swinunetr/splits_10patients.json').read_text())[0]
    case = split['train'][0]
    frame = core.base._load_pkl_dataframe(STAGE/'data/MNI.pkl')
    item = core.PreparedSubfields(frame, [case], DATASETS['MNI'])[0]
    label_tensor = item['label'][None]
    inner_tensor, outer_tensor = build_boundary_bands(label_tensor > 0, steps=2)
    image, labels = item['image'][0].numpy(), item['label'][0].numpy()
    inner, outer = inner_tensor[0,0].numpy(), outer_tensor[0,0].numpy()
    assert not np.any(inner & outer)
    assert np.all(labels[inner] > 0) and np.all(labels[outer] == 0)
    foreground = labels > 0
    # Choose a large, interpretable cross-section using labels only.
    candidates=[]
    for axis in range(3):
        areas=foreground.sum(axis=tuple(i for i in range(3) if i!=axis))
        candidates.append((int(areas.max()),axis,int(areas.argmax())))
    _,axis,index=max(candidates)
    plane=lambda array:np.take(array,index,axis=axis)
    support=plane(inner|outer)
    coordinates=np.argwhere(support)
    lo=np.maximum(0,coordinates.min(0)-5)
    hi=np.minimum(support.shape,coordinates.max(0)+6)
    crop=(slice(lo[0],hi[0]),slice(lo[1],hi[1]))
    display=lambda array:plane(array)[crop].T
    img, lab, ib, ob=map(display,(image,labels,inner,outer))
    inside=display(foreground)
    intensity=image[image!=0]
    vmin,vmax=np.percentile(intensity,[1,99])
    label_rgba=np.zeros((*lab.shape,4)); union_rgba=np.zeros_like(label_rgba); aware_rgba=np.zeros_like(label_rgba)
    for c,color in COLORS.items():
        label_rgba[lab==c]=to_rgba(color,.65)
        aware_rgba[ib & (lab==c)]=to_rgba(color,.8)
    union_rgba[ib]=to_rgba(INNER,.8)
    union_rgba[ob]=to_rgba(OUTER,.8)
    aware_rgba[ob]=to_rgba(OUTER,.8)
    fig,axes=plt.subplots(2,2,figsize=(11,9.4),facecolor='white')
    panels=[('MRI',None),('Ground-truth subfields',label_rgba),
            ('Original bands: foreground / background',union_rgba),
            ('Class-aware bands: correct subfield / background',aware_rgba)]
    for ax,(title,overlay) in zip(axes.flat,panels):
        ax.imshow(img,cmap='gray',vmin=vmin,vmax=vmax,origin='lower',interpolation='nearest')
        if overlay is not None:ax.imshow(overlay,origin='lower',interpolation='nearest')
        if overlay is not None:ax.contour(inside.astype(float),levels=[.5],colors='white',linewidths=.65)
        ax.set_title(title,fontsize=11,pad=9)
        ax.set_xticks([]);ax.set_yticks([])
    native_index=index-(64-DATASETS['MNI']['native_shape'][axis])//2
    fig.suptitle('Where bands supervise an actual MNI training example',fontsize=17,y=.98)
    fig.text(.5,.944,f'Subject 03 · left hippocampus · voxel-axis {axis}, native slice {native_index} · no resizing',ha='center',fontsize=10)
    legends=[Patch(facecolor=COLORS[c],label=name) for c,name in [(1,'CA1–3'),(2,'Subiculum'),(3,'CA4–DG')]]
    legends += [Patch(facecolor=INNER,label='Inner band: any foreground'),Patch(facecolor=OUTER,label='Outer band: background')]
    fig.legend(handles=legends,loc='lower center',bbox_to_anchor=(.5,.055),ncol=3,frameon=False,fontsize=10)
    fig.text(.5,.028,'White line: true outer contour. Both variants use the same 3D, two-step band locations.',ha='center',fontsize=10)
    fig.text(.5,.009,'Uncolored voxels receive no bands loss; the base segmentation loss still applies throughout the image.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.045,right=.955,top=.90,bottom=.15,wspace=.08,hspace=.18)
    fig.savefig(OUT/'bands_overlay_example.png',dpi=180)
    fig.savefig(OUT/'bands_overlay_example.pdf')
    plt.close(fig)
    counts={str(c):dict(total_voxels=int((labels==c).sum()),inner_band_voxels=int((inner & (labels==c)).sum())) for c in (1,2,3)}
    metadata=dict(case=case,split='fold0 training',axis=axis,padded_slice=index,native_slice=native_index,
        selection='First training crop; largest foreground cross-section across voxel axes, label-only selection.',
        band_geometry='Exact frozen build_boundary_bands: two iterations, six-connected 3D erosion/dilation on foreground union.',
        source_pkl_sha256=digest(STAGE/'data/MNI.pkl'),
        band_source_sha256=digest(Path(__import__('thesis.new_constraints.bands.outer_boundary',fromlist=['x']).__file__)),
        counts=counts,inner_voxels=int(inner.sum()),outer_voxels=int(outer.sum()),
        slice_counts=dict(inner=int(ib.sum()),outer=int(ob.sum())),
        interpretation='Training-label geometry, not model prediction/error map. Displayed in source voxel axes; anatomical orientation is not independently verified. Same native voxels, symmetric training padding and ROI zoom only.')
    (OUT/'bands_overlay_example.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
