"""Prepare aligned ADNI hippocampus crops from local study files."""
import argparse
import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_all_labels(dir, name=''):
    files = glob.glob(os.path.join(dir, name))

    aggreg = None

    for f in files:
        if "CSF" in f:
            continue
        img = nib.load(f)
        data = img.get_fdata()
        if aggreg is None:
            aggreg = data.copy()
        else:
            aggreg += data

    aggreg = np.array(aggreg)
    print(aggreg.shape)

    # crop to leave out the zeros
    # find the min-max indices along each axis

    def find_min_max(single, axis=0):
        min_v = None
        for i in range(single.shape[0]):
            idx = [slice(None)] * 3
            idx[axis] = i

            if np.any(single[tuple(idx)]):
                min_v = i
                break

        max_v = None
        for i in range(single.shape[axis] - 1, -1, -1):
            idx = [slice(None)] * 3
            idx[axis] = i
            if np.any(single[tuple(idx)]):
                max_v = i
                break

        return min_v, max_v

    # save to disk a numpy file of all crops
    import pandas as pd

    files = [f for f in files if 'CSF' not in f]  # exclude CSF images

    data_list = []
    crop_idxs = []

    half_x = 32 // 2
    half_y = 48 // 2
    half_z = 40 // 2

    for f in files:
        img = nib.load(f)
        data = img.get_fdata()
        # cropped = data[min_x:max_x + 1, min_y:max_y + 1, min_z:max_z + 1]

        min_x, max_x = find_min_max(data, axis=0)
        min_y, max_y = find_min_max(data, axis=1)
        min_z, max_z = find_min_max(data, axis=2)

        print(f"Cropping indices: x({min_x},{max_x}), y({min_y},{max_y}), z({min_z},{max_z})")

        # calculate mid points
        avg_x = (min_x + max_x) // 2
        avg_y = (min_y + max_y) // 2
        avg_z = (min_z + max_z) // 2

        crop_idx = ((avg_x - half_x, avg_x + half_x),
                    (avg_y - half_y, avg_y + half_y),
                    (avg_z - half_z, avg_z + half_z))

        cropped = data[avg_x - half_x:avg_x + half_x,
                  avg_y - half_y:avg_y + half_y,
                  avg_z - half_z:avg_z + half_z].copy()

        # subsample 2x
        # cropped = cropped[::2, ::2, ::2]

        data_list.append(cropped)
        crop_idxs.append(crop_idx)

    # now subsample all images and labels 2x

    # a filename has this structure:
    # ADNI_nnn_S_nnnn_xxxxx_D.nii
    # the subject id is nnn_S_nnnn
    direction = 'R' if '_R.mnc' in name else 'L'
    metadata = [os.path.basename(f).split('_' + direction + '.mnc')[0] for f in files]
    metadata = [sid.replace('ADNI_', '') for sid in metadata]
    metadata = [("_".join(sid.split('_')[:-1]), sid.split('_')[-1]) for sid in metadata]
    subject_ids = [sid for sid, _ in metadata]
    image_ids = [img_id for _, img_id in metadata]


    df = pd.DataFrame({'data': data_list, 'filename': files, 'subject_id': subject_ids, 'image_id': image_ids,
                       'direction': [direction] * len(files), 'crop_idx': crop_idxs})

    # data = np.array(data)
    # assert data.shape[1:] == (2 * half, 2 * half, 2 * half), f"Unexpected shape: {data.shape}"
    # save to disk
    # name = name.replace('*', '')
    # print(name, data.shape, f"x({min_x},{max_x}), y({min_y},{max_y}), z({min_z},{max_z})")
    # np.savez_compressed(name + '.npz', data)
    # return data

    return df


def merge_image_data_to_label_data(df, dir):
    print('merge start')
    files = glob.glob(os.path.join(dir, '*/*.mnc'))

    image_data_dict = {}
    for f in tqdm(files):
        if "CSF" in f:
            continue
        img = nib.load(f)
        data = img.get_fdata()
        filename = os.path.basename(f)
        subject_id = "_".join(("_".join(filename.split('_')[:5])).split('_')[1:4])
        image_id = filename.split('_')[4]
        key = (subject_id, image_id)
        image_data_dict[key] = data
        print(f'Loaded {filename} for subject_id={subject_id}, image_id={image_id}')
        del img

    # now merge
    image_data_list = []
    for idx, row in tqdm(df.iterrows()):
        image_id = row['image_id']
        subject_id = row['subject_id']
        key = (subject_id, image_id)
        image_data = image_data_dict.get(key, None)
        if image_data is None:
            print(f"Warning: image data not found for subject_id={subject_id}, image_id={image_id}")
        else:
            crop_idx = row['crop_idx']
            cropped_image_data = image_data[
                                 crop_idx[0][0]:crop_idx[0][1],
                                 crop_idx[1][0]:crop_idx[1][1],
                                 crop_idx[2][0]:crop_idx[2][1]
                                 ]
            # subsample 2x
            # cropped_image_data = cropped_image_data[::2, ::2, ::2]
            image_data_list.append(cropped_image_data)

    df['image_data'] = image_data_list
    print(f'Created image_data for {len(image_data_list)} entries')
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True, help='Extracted directory containing Labels* folders')
    parser.add_argument('--images', type=Path, required=True, help='Extracted ACPC MINC scan directory')
    args = parser.parse_args()
    for folder in ('imagesTr', 'labelsTr'):
        (args.target / folder).mkdir(parents=True, exist_ok=True)
    df_L = get_all_labels(str(args.labels / 'Labels*'), '*_L.mnc')
    df_R = get_all_labels(str(args.labels / 'Labels*'), '*_R.mnc')
    df = pd.concat([df_L, df_R], ignore_index=True)
    merge_image_data_to_label_data(df, dir=str(args.images))
    df.to_pickle(os.path.join(args.target, 'adni_hippocampus_full.pkl'), compression="gzip")

    identity_affine = np.eye(4)

    for index, row in df.iterrows():
        image_id = row['image_id']
        direction = row['direction']  # Get the direction (L or R)
        image_data = row['image_data']
        label_data = row['data']

        # Create NIfTI image for image_data with direction in filename
        image_nifti = nib.Nifti1Image(image_data, identity_affine)
        image_filename_dataset = os.path.join(args.target, 'imagesTr',
                                              f'hippocampus_adni_{image_id}_{direction}_0000.nii.gz')
        nib.save(image_nifti, image_filename_dataset)

        # Create NIfTI image for label_data with direction in filename
        label_nifti = nib.Nifti1Image(label_data, identity_affine)
        label_filename_dataset = os.path.join(args.target, 'labelsTr',
                                              f'hippocampus_adni_{image_id}_{direction}.nii.gz')
        nib.save(label_nifti, label_filename_dataset)

    print(f"Successfully saved {len(df)} image and label files.")


if __name__ == "__main__":
    main()
