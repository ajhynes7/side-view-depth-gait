"""Extract ground truth positions from labelled images."""

from collections import OrderedDict
import glob
import os
from os.path import basename, join
import pickle
import re

import cv2
import numpy as np
import pandas as pd

import analysis.images as im


def main():

    load_dir = join('data', 'kinect', 'labelled_trials')
    align_dir = join('data', 'kinect', 'alignment')

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

    part_names, rgb_vectors = zip(*part_rgb_dict.items())

    # Regex to extract frame number from file name
    pattern = re.compile(r'(\d+)\.png')

    dict_truth = {}

    for trial_name in labelled_trial_names:

        label_dir = join(load_dir, trial_name, 'label')
        depth_dir = join(load_dir, trial_name, 'depth16bit')

        label_paths = sorted(glob.glob(join(label_dir, '*.png')))
        depth_paths = sorted(glob.glob(join(depth_dir, '*.png')))

        depth_filenames = [basename(x) for x in depth_paths]
        image_nums = [
            int(re.search(pattern, x).group(1)) for x in depth_filenames
        ]

        df_trial = pd.DataFrame(index=image_nums, columns=part_names)

        # %% Iterate through labelled images for walking trial

        for i, image_num in enumerate(image_nums):

            label_path, depth_path = label_paths[i], depth_paths[i]

            label_image_rgb = cv2.imread(label_path, cv2.IMREAD_ANYCOLOR)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 10

            label_image = im.rgb_to_label(label_image_rgb, rgb_vectors)

            for j, part_name in enumerate(part_rgb_dict):

                # Binary image of one body part
                label = j + 1
                part_binary = label_image == label

                if not np.any(part_binary):
                    # There are no pixels for this part in the label image
                    continue

                nonzero_row_col = np.argwhere(part_binary)
                depths = depth_image[part_binary]

                image_points = np.column_stack((nonzero_row_col[:, 1],
                                                nonzero_row_col[:, 0], depths))

                median_image = np.median(image_points, axis=0)
                median_real = im.image_to_real(median_image, im.X_RES,
                                               im.Y_RES, im.F_XZ, im.F_YZ)

                df_trial.loc[image_num, part_name] = median_real

        # Load dictionary to convert image numbers to frames
        with open(join(align_dir, "{}.pkl".format(trial_name)),
                  'rb') as handle:
            image_to_frame = pickle.load(handle)

        df_trial.index = df_trial.index.map(image_to_frame)

        df_trial = df_trial.dropna(how='all')
        dict_truth[trial_name] = df_trial

    # True positions from all labelled trials
    df_truth = pd.concat(dict_truth)
    df_truth.index.names = ['trial_name', 'frame']

    df_truth.to_pickle(join('results', 'dataframes', 'df_truth.pkl'))


if __name__ == '__main__':
    main()
