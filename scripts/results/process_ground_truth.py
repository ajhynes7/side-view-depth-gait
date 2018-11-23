"""Extract ground truth positions from labelled images."""

import glob
import os
import re
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

import analysis.images as im


def main():

    load_dir = os.path.join('data', 'kinect', 'labelled_trials')

    # Camera calibration parameters
    x_res, y_res = 565, 430
    f_xz, f_yz = 1.11146664619446, 0.833599984645844

    part_rgb_dict = OrderedDict({
        'HEAD': [255, 0, 255],
        'L_HIP': [0, 62, 192],
        'R_HIP': [192, 0, 62],
        'L_THIGH': [192, 126, 126],
        'R_THIGH': [126, 255, 255],
        'L_KNEE': [126, 0, 126],
        'R_KNEE': [0, 126, 126],
        'L_CALF': [0, 62, 0],
        'R_CALF': [0, 0, 126],
        'L_FOOT': [0, 230, 0],
        'R_FOOT': [0, 0, 255],
    })

    labelled_trial_names = os.listdir(load_dir)

    # Regex to extract frame number from file name
    pattern = re.compile(r'(\d+)\.png')

    part_names, rgb_vectors = zip(*part_rgb_dict.items())

    dict_truth = {}

    for trial_name in labelled_trial_names:

        label_dir = os.path.join(load_dir, trial_name, 'label')
        depth_dir = os.path.join(load_dir, trial_name, 'depth16bit')

        label_paths = sorted(glob.glob(os.path.join(label_dir, '*.png')))
        depth_paths = sorted(glob.glob(os.path.join(depth_dir, '*.png')))

        depth_filenames = [os.path.basename(x) for x in depth_paths]
        frames = [int(re.search(pattern, x).group(1)) for x in depth_filenames]

        df_trial = pd.DataFrame(index=frames, columns=part_names)

        for ii, frame in enumerate(frames):

            label_path, depth_path = label_paths[ii], depth_paths[ii]

            label_image_rgb = cv2.imread(label_path, cv2.IMREAD_ANYCOLOR)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 10

            label_image = im.rgb_to_label(label_image_rgb, rgb_vectors)

            for i, part_name in enumerate(part_rgb_dict):

                # Binary image of one body part
                label = i + 1
                part_binary = label_image == label

                if not np.any(part_binary):
                    # There are no pixels for this part in the label image
                    continue

                nonzero_row_col = np.argwhere(part_binary)
                depths = depth_image[part_binary]

                image_points = np.column_stack(
                    (nonzero_row_col[:, 1], nonzero_row_col[:, 0], depths))

                median_image = np.median(image_points, axis=0)
                median_real = im.image_to_real(
                    median_image, x_res, y_res, f_xz, f_yz)

                df_trial.loc[frame, part_name] = median_real

        dict_truth[trial_name] = df_trial

        # True positions from all labelled trials
        df_truth = pd.concat(dict_truth)
        df_truth.index.names = ['trial_name', 'frame']

        df_truth.to_pickle(
            os.path.join('results', 'dataframes', 'df_truth.pkl'))


if __name__ == '__main__':
    main()
