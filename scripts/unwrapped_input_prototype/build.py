"""Create the MRI geometry prototype and its local, interactive inspection page."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/hippo-input-prototype-mpl')
sys.path.insert(0, '/private/tmp/hippo-unwrap-deps')
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from PIL import Image
from skimage.measure import marching_cubes
from plotly.offline import get_plotlyjs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .geometry import fit_straightening, matched_axes, sample_world, sample_grid, inside_grid
from scripts.depth_supervision.geometry import world_points

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evaluation_output/unwrapped_input_prototype'
CASES = ['hippocampus_164', 'hippocampus_017', 'hippocampus_330']
PRED_SHA = '922980d1c1b29bf8dd7a78e0643785b8b54ea52be5e23988b37650fe9fa5edb0'
LABEL_COLORS = {1:(237,158,53), 2:(58,163,232)}
ERROR_COLORS = {1:(237,92,151), 2:(47,195,177), 3:(168,129,238)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def padded(array):
    assert all(s <= 64 for s in array.shape)
    widths = [((64-s)//2, 64-s-(64-s)//2) for s in array.shape]
    return np.pad(array, widths), tuple(slice(a, a+s) for (a,_),s in zip(widths,array.shape))


def png(array):
    stream=io.BytesIO()
    Image.fromarray(array).save(stream,format='PNG',optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()


def gray_frame(values, valid, low, high):
    gray=np.round(np.clip((values-low)/(high-low),0,1)*255).astype(np.uint8)
    rgb=np.repeat(gray[...,None],3,axis=-1)
    rgb[~valid]=[16,22,31]
    return np.flip(rgb.transpose(1,0,2),axis=0)


def overlay_frame(labels, colors, contour=False):
    rgba=np.zeros((*labels.shape,4),np.uint8)
    for c,color in colors.items():
        mask=labels==c
        if contour:
            mask=mask & ~ndi.binary_erosion(mask)
        rgba[mask,:3]=color;rgba[mask,3]=230 if contour else 190
    return np.flip(rgba.transpose(1,0,2),axis=0)


def errors(gt,pred):
    result=np.zeros(gt.shape,np.uint8)
    result[(gt>0)&(pred==0)]=1
    result[(gt==0)&(pred>0)]=2
    result[(gt>0)&(pred>0)&(gt!=pred)]=3
    return result


def intensity_metrics(original,reconstructed,mask,valid,scale):
    used=mask & valid
    if not used.any():
        return dict(voxels=int(mask.sum()),covered_voxels=0,coverage=0.,nrmse_percent=None)
    delta=reconstructed[used]-original[used]
    return dict(voxels=int(mask.sum()),covered_voxels=int(used.sum()),coverage=float(used.sum()/mask.sum()),
                nrmse_percent=float(100*np.sqrt(np.mean(delta**2))/scale),
                mean_absolute_percent=float(100*np.mean(np.abs(delta))/scale),
                p95_absolute_percent=float(100*np.percentile(np.abs(delta),95)/scale))


def mesh_data(pred,affine,transform):
    meshes=[]
    for label in (1,2):
        verts,faces,_,_=marching_cubes((pred==label).astype(np.float32),.5,step_size=1)
        xyz=world_points(verts,affine)
        meshes.append(dict(label=label,color='#ed9e35' if label==1 else '#3aa3e8',
                           original=np.round(transform.world_to_local(xyz),4).tolist(),
                           straight=np.round(transform.world_to_straight(xyz),4).tolist(),faces=faces.tolist()))
    s=np.linspace(transform.knots[0],transform.knots[-1],120)
    return dict(meshes=meshes,guide_original=np.c_[s,transform.offset(s),np.zeros_like(s)].tolist(),
                guide_straight=np.c_[s,np.zeros_like(s),np.zeros_like(s)].tolist())


def make_case(case,pred,protocol):
    directory=OUT/case;directory.mkdir(exist_ok=True)
    image_path=ROOT/'datasets/Dataset101_MSD/imagesTr'/(case+'_0000.nii.gz')
    label_path=ROOT/'datasets/Dataset101_MSD/labelsTr'/(case+'.nii.gz')
    image_file=nib.load(image_path)
    image_native=np.asarray(image_file.dataobj,dtype=np.float32)
    raw,native_slices=padded(image_native)
    fov,_=padded(np.ones_like(image_native,dtype=np.float32))
    affine=np.asarray(protocol['native_geometry'][case]['model_affine'])
    mu=float(image_native[image_native!=0].mean());sigma=float(image_native[image_native!=0].std())
    image=np.where(raw!=0,(raw-mu)/sigma,0.).astype(np.float32)
    # The complete input mapping is constructed before the annotation is read.
    transform=fit_straightening(pred,affine)
    vox=np.indices(pred.shape).reshape(3,-1).T
    xyz=world_points(vox,affine)
    roi=ndi.distance_transform_edt(pred==0,sampling=np.linalg.norm(affine[:3,:3],axis=0))<=8.
    axes=matched_axes(transform,xyz[roi.ravel()])
    q=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
    control_world=transform.local_to_world(q)
    straight_world=transform.straight_to_world(q)
    control=sample_world(image,affine,control_world)
    straight=sample_world(image,affine,straight_world)
    control_valid=sample_world(fov,affine,control_world)>=1-1e-6
    straight_valid=sample_world(fov,affine,straight_world)>=1-1e-6
    native_q=transform.world_to_straight(xyz).reshape((*pred.shape,3))
    native_local=transform.world_to_local(xyz).reshape((*pred.shape,3))
    reconstructed=sample_grid(straight,axes,native_q)
    control_reconstructed=sample_grid(control,axes,native_local)
    reconstructed_valid=(sample_grid(straight_valid.astype(np.float32),axes,native_q)>=1-1e-5)&inside_grid(axes,native_q)
    control_reconstructed_valid=(sample_grid(control_valid.astype(np.float32),axes,native_local)>=1-1e-5)&inside_grid(axes,native_local)
    # For display, return directly to the original plane-aligned sampling grid.
    return_q=transform.world_to_straight(control_world)
    back=sample_grid(straight,axes,return_q)
    back_valid=(sample_grid(straight_valid.astype(np.float32),axes,return_q)>=1-1e-5)&control_valid
    gt_native=np.asarray(nib.load(label_path).dataobj).astype(np.uint8)
    assert hashlib.sha256(gt_native.tobytes()).hexdigest()==protocol['native_geometry'][case]['label_array_sha256']
    gt,_=padded(gt_native)
    low,high=np.percentile(image[(fov>0)&(raw!=0)],[1,99]).tolist()
    scale=high-low
    roundtrip=transform.straight_to_world(transform.world_to_straight(xyz))
    inverse_error=np.linalg.norm(roundtrip-xyz,axis=1)
    sample_s=np.linspace(axes[0][0],axes[0][-1],500)
    slopes=transform.offset(sample_s,1)
    jacobians=np.repeat(np.eye(3)[None],len(slopes),axis=0);jacobians[:,1,0]=slopes
    conditions=np.linalg.cond(jacobians)
    # Independent numerical finite differences verify determinant and inverse.
    testq=q[::8,::8,::8].reshape(-1,3);eps=1e-3
    jac=np.stack([(transform.straight_to_world(testq+eps*np.eye(3)[i])-transform.straight_to_world(testq-eps*np.eye(3)[i]))/(2*eps) for i in range(3)],axis=-1)
    determinants=np.linalg.det(jac)
    masks={'predicted_tissue':pred>0,'annotated_tissue':gt>0,
           'missed_tissue':(gt>0)&(pred==0),'prediction_plus_8mm_in_native_fov':roi&(fov>0)}
    measurement={}
    for name,mask in masks.items():
        comparison_valid=reconstructed_valid & control_reconstructed_valid
        measurement[name]=dict(straight=intensity_metrics(image,reconstructed,mask,reconstructed_valid,scale),
                               plane_control=intensity_metrics(image,control_reconstructed,mask,control_reconstructed_valid,scale),
                               matched_support_straight=intensity_metrics(image,reconstructed,mask,comparison_valid,scale),
                               matched_support_control=intensity_metrics(image,control_reconstructed,mask,comparison_valid,scale),
                               coordinate_grid_coverage=float(inside_grid(axes,native_q)[mask].mean()))
    meta=transform.metadata()
    audit=dict(case=case,geometry_source='full-data depth-only SwinUNETR, best epoch 12; foreground union only',
        annotation_used_to_build_input=False,transform=meta,
        grid_shape=list(straight.shape),grid_spacing_mm=[float(a[1]-a[0]) for a in axes],
        grid_axes_ranges_mm=[[float(a[0]),float(a[-1])] for a in axes],
        maximum_coordinate_roundtrip_error_mm=float(inverse_error.max()),
        numerical_jacobian_min=float(determinants.min()),numerical_jacobian_max=float(determinants.max()),
        maximum_local_condition_number=float(conditions.max()),
        maximum_transverse_shift_mm=float(np.max(np.abs(transform.offset(sample_s)))),
        interpolation='trilinear MRI, nearest-neighbour labels for QA overlays only',
        normalization=dict(nonzero_mean=mu,nonzero_std=sigma,display_p1=low,display_p99=high),
        intensity_metrics=measurement,
        original_prediction_errors={k:int((errors(gt,pred)==i).sum()) for k,i in [('missed',1),('extra',2),('ap_swaps',3)]},
        image_sha256=digest(image_path),annotation_sha256=digest(label_path),prediction_archive_sha256=PRED_SHA,
        limitations=['Geometric straightening, not cortical/laminar unfolding.',
                     'Full-data model geometry only; not valid as a 5% data experiment input.',
                     'Perfect coordinate inversion does not imply lossless MRI interpolation.',
                     'Transverse shear preserves volume but changes local angles and distances.',
                     'Invalid or out-of-FOV samples have an explicit validity mask.'])
    assert inverse_error.max()<1e-8 and determinants.min()>.999999
    assert measurement['predicted_tissue']['coordinate_grid_coverage']==1.
    (directory/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (directory/'transform.json').write_text(json.dumps(meta,indent=2)+'\n')
    np.savez_compressed(directory/'volumes_and_maps.npz',image_native=image,image_control=control,image_straight=straight,
        image_roundtrip_native=reconstructed,image_control_roundtrip_native=control_reconstructed,
        control_valid=control_valid,straight_valid=straight_valid,roundtrip_valid_native=reconstructed_valid,
        control_roundtrip_valid_native=control_reconstructed_valid,
        straight_grid_to_world_mm=straight_world.astype(np.float32),native_to_straight_mm=native_q.astype(np.float32),
        s_mm=axes[0],t_mm=axes[1],signed_depth_mm=axes[2],model_affine=affine,
        prediction_native=pred,annotation_native_qa_only=gt)
    nib.save(nib.Nifti1Image(reconstructed[native_slices],image_file.affine),directory/'mri_roundtrip_native_zscore.nii.gz')
    nib.save(nib.Nifti1Image(reconstructed_valid[native_slices].astype(np.uint8),image_file.affine),directory/'roundtrip_valid_native.nii.gz')
    control_pred=sample_world(pred,affine,control_world,order=0).astype(np.uint8)
    straight_pred=sample_world(pred,affine,straight_world,order=0).astype(np.uint8)
    control_gt=sample_world(gt,affine,control_world,order=0).astype(np.uint8)
    straight_gt=sample_world(gt,affine,straight_world,order=0).astype(np.uint8)
    difference=100*np.abs(back-control)/scale
    # Crop the display only; saved network-input candidates remain full 64^3 grids.
    display_points=np.concatenate([transform.world_to_local(xyz[(pred>0).ravel()]),transform.world_to_straight(xyz[(pred>0).ravel()])])
    lower,upper=display_points[:,:2].min(0)-8,display_points[:,:2].max(0)+8
    crop=tuple(slice(max(0,int(np.searchsorted(axes[i],lower[i]))),min(64,int(np.searchsorted(axes[i],upper[i]))+1)) for i in range(2))
    frames=[]
    for k in range(len(axes[2])):
        plane=(*crop,k)
        a,b,c=control[plane],straight[plane],back[plane]
        v=[control_valid[plane],straight_valid[plane],back_valid[plane]]
        panels=[png(gray_frame(z,valid,low,high)) for z,valid in zip([a,b,c],v)]
        heat=(plt.get_cmap('magma')(np.clip(difference[plane]/20,0,1))[...,:3]*255).astype(np.uint8)
        heat[~back_valid[plane]]=[16,22,31]
        panels.append(png(np.flip(heat.transpose(1,0,2),axis=0)))
        overlays={}
        for key,labels,colors,contour in [('prediction',[control_pred[plane],straight_pred[plane],control_pred[plane]],LABEL_COLORS,True),
            ('annotation',[control_gt[plane],straight_gt[plane],control_gt[plane]],LABEL_COLORS,True),
            ('errors',[errors(control_gt[plane],control_pred[plane]),errors(straight_gt[plane],straight_pred[plane]),errors(control_gt[plane],control_pred[plane])],ERROR_COLORS,False)]:
            overlays[key]=[png(overlay_frame(z,colors,contour)) for z in labels]
        frames.append(dict(images=panels,overlays=overlays,predicted_voxels=int((control_pred[:,:,k]>0).sum()),
                           annotated_voxels=int((control_gt[:,:,k]>0).sum())))
    data=dict(case=case,audit=audit,depths=np.round(axes[2],3).tolist(),initial_depth=int(np.argmin(np.abs(axes[2]))),
              frames=frames,geometry=mesh_data(pred,affine,transform),download=f'{case}/volumes_and_maps.npz',
              display_extent_mm=[[float(axes[i][crop[i].start]),float(axes[i][crop[i].stop-1])] for i in range(2)])
    k=data['initial_depth'];extent=[axes[0][crop[0].start],axes[0][crop[0].stop-1],axes[1][crop[1].start],axes[1][crop[1].stop-1]]
    fig,axs=plt.subplots(2,3,figsize=(15,9),facecolor='#f4f6fa')
    for ax,z,title in zip(axs[0],[control,straight,back],['Original MRI · plane-aligned','Straightened MRI','Mapped back to the original plane']):
        ax.imshow(z[(*crop,k)].T,origin='lower',extent=extent,cmap='gray',vmin=low,vmax=high)
        ax.set_title(title);ax.set_xlabel('Longitudinal position (mm)');ax.set_ylabel('Transverse position (mm)')
    ax=axs[1,0]
    im=ax.imshow(np.where(back_valid[(*crop,k)],difference[(*crop,k)],np.nan).T,origin='lower',extent=extent,cmap='magma',vmin=0,vmax=20)
    ax.set_title('MRI interpolation difference');fig.colorbar(im,ax=ax,label='% of MRI display range')
    fg=xyz[(pred>0).ravel()]
    for index,points,title in [(1,transform.world_to_local(fg),'Predicted volume · original geometry'),(2,transform.world_to_straight(fg),'Predicted volume · straightened geometry')]:
        axs[1,index].remove();ax=fig.add_subplot(2,3,4+index,projection='3d')
        colors=np.where(pred[pred>0]==1,'#ed9e35','#3aa3e8')
        ax.scatter(*points[::2].T,c=colors[::2],s=3,alpha=.4,rasterized=True)
        line=np.c_[sample_s,transform.offset(sample_s) if index==1 else np.zeros_like(sample_s),np.zeros_like(sample_s)]
        keep=(sample_s>=transform.knots[0])&(sample_s<=transform.knots[-1]);ax.plot(*line[keep].T,c='#152e47',lw=2)
        ax.set_title(title);ax.set_xlabel('Longitudinal');ax.set_ylabel('Transverse');ax.set_zlabel('Signed depth')
        ax.view_init(elev=23,azim=-64)
    fig.suptitle(f'{case} · prediction-guided MRI input prototype\nDepth {axes[2][k]:+.1f} mm · geometry uses no annotation',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(directory/'overview.png',dpi=160);plt.close(fig)
    print(json.dumps({k:audit[k] for k in ['case','maximum_coordinate_roundtrip_error_mm','numerical_jacobian_min','maximum_local_condition_number','maximum_transverse_shift_mm','intensity_metrics']}),flush=True)
    return data


def main():
    OUT.mkdir(exist_ok=True)
    archive=OUT/'source/validation_predictions_full_depth.npz'
    assert digest(archive)==PRED_SHA
    snapshot=json.loads((ROOT/'evaluation_output/depth_swinunetr_20261006/full_data_completed_snapshot.json').read_text())
    protocol=snapshot['runs']['depth']['config']['protocol']
    with np.load(archive) as predictions:
        cases=[make_case(case,predictions[case],protocol) for case in CASES]
    (OUT/'data.json').write_text(json.dumps(cases,separators=(',',':')))
    (OUT/'plotly.min.js').write_text(get_plotlyjs())
    template=Path(__file__).with_name('viewer.html').read_text()
    (OUT/'index.html').write_text(template)
    (OUT/'prototype_audit.json').write_text(json.dumps([c['audit'] for c in cases],indent=2)+'\n')
    with zipfile.ZipFile(OUT/'prototype_data.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.write(OUT/'prototype_audit.json','prototype_audit.json')
        for case in CASES:
            for name in ['volumes_and_maps.npz','transform.json','audit.json','mri_roundtrip_native_zscore.nii.gz','roundtrip_valid_native.nii.gz','overview.png']:
                z.write(OUT/case/name,f'{case}/{name}')
    print('Prototype ready:',OUT/'index.html',flush=True)


if __name__ == '__main__':
    main()
