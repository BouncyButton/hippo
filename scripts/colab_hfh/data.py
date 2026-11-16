"""Frozen localized binary HFH crops; no augmentation or label-derived prompts."""
from pathlib import Path
import os
import numpy as np


class Volumes:
    def __init__(self, protocol):
        self.root = Path(protocol['data'])

    def load(self, case, labels=True):
        from evaluation.medsam3_zero_shot import normalize_image
        with np.load(self.root/(case+'.npz'), allow_pickle=False) as item:
            # Original array axes X,Y,Z -> X,Z,Y: preserve 2 mm slice spacing.
            image = normalize_image(item['image'].transpose(0, 2, 1).copy())
            truth = item['label'].transpose(0, 2, 1).copy() if labels else None
        return image, truth


class PreparedPatients:
    def __init__(self, descriptor, names):
        import torch
        from monai.transforms import NormalizeIntensity
        self.items = []
        for name in names:
            with np.load(Path(descriptor['data'])/(name+'.npz'), allow_pickle=False) as item:
                image, label = item['image'].copy(), item['label'].copy()
            if image.shape != (64, 32, 64) or label.shape != image.shape:
                raise ValueError('Unexpected HFH crop geometry')
            image = np.asarray(NormalizeIntensity(nonzero=True, channel_wise=True)(image[None]))
            image = np.pad(image, ((0, 0), (0, 0), (16, 16), (0, 0)))
            label = np.pad(label, ((0, 0), (16, 16), (0, 0)))
            item = dict(image=torch.from_numpy(image).float(), label=torch.from_numpy(label[None]).long(),
                        case_name=name, subject_id=name.rsplit('_', 1)[0])
            if os.environ.get('HIPPO_HFH_GPU_CACHE') == '1':
                for key in ('image', 'label'):
                    cached = item[key].cuda()
                    if not torch.equal(cached.cpu(), item[key]):
                        raise ValueError('GPU resident input cache is not exact')
                    item[key] = cached
            self.items.append(item)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]
