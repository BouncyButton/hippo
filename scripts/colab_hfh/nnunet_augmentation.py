"""Small, paired HFH augmentations with an independent checkpointed CPU RNG."""
import hashlib

NAME = 'hfh_mild_shift_intensity_v1'
FLOPS_PER_UPDATE = 128 * 64**3
POLICY = dict(name=NAME, identity_probability=.25, shift_probability=.5,
              shift_voxels_each_inplane_axis=[-1, 0, 1], thick_axis_shift=0,
              contrast_probability=.5, contrast_range=[.95, 1.05],
              noise_probability=.25, noise_std=.01,
              intensity_units='Nonzero-normalized MRI standard deviations',
              spatial='Integer translation only; no interpolation, reflection or deformation',
              foreground_guard='Reject translation if any foreground voxel is lost',
              geometry='Recompute physical bands from each transformed label; spacing unchanged',
              rng='Checkpointed CPU augmentation_rng; independent of model and sample order',
              evaluation='Original unaugmented training and validation crops',
              charged_flops_per_update=FLOPS_PER_UPDATE)


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def shift_pair(image, label, shifts):
    """Translate without wrapping content; preserve exact integer annotations."""
    import torch
    if shifts[1] != 0 or any(abs(v)>1 for v in shifts):
        raise ValueError('Only one-voxel in-plane translations are authorized')
    shifted_image=torch.roll(image,shifts=shifts,dims=(2,3,4))
    shifted_label=torch.roll(label,shifts=shifts,dims=(2,3,4))
    for axis,delta in zip((2,3,4),shifts):
        if not delta:continue
        region=[slice(None)]*5
        region[axis]=slice(0,delta) if delta>0 else slice(delta,None)
        shifted_image[tuple(region)]=0
        shifted_label[tuple(region)]=0
    if int(shifted_label.sum())!=int(label.sum()):
        return image,label,False
    return shifted_image,shifted_label,True


def augment(image,label,generator):
    import torch
    if tuple(image.shape)!=(1,1,64,64,64) or label.shape!=image.shape:
        raise ValueError('Expected frozen HFH grid')
    before=tensor_sha(generator.get_state())
    u=torch.rand(8,generator=generator).tolist()
    record=dict(identity=u[0]<.25,shift=[0,0,0],shift_rejected=False,
                contrast_factor=1.,noise_std=0.,rng_before_sha256=before)
    if not record['identity']:
        if u[1]<.5:
            shifts=(min(2,int(3*u[2]))-1,0,min(2,int(3*u[3]))-1)
            image,label,accepted=shift_pair(image,label,shifts)
            record.update(shift=list(shifts) if accepted else [0,0,0],shift_rejected=not accepted)
        support=image!=0
        output=image.float()
        if u[4]<.5:
            record['contrast_factor']=.95+.1*u[5]
            output=output*record['contrast_factor']
        if u[6]<.25:
            record['noise_std']=.01
            noise=torch.randn(image.shape,generator=generator,dtype=torch.float32).to(image.device)
            output=output+.01*noise
        image=torch.where(support,output,torch.zeros_like(output)).to(image.dtype)
    record.update(rng_after_sha256=tensor_sha(generator.get_state()),
                  image_sha256=tensor_sha(image),label_sha256=tensor_sha(label))
    return image,label,record
