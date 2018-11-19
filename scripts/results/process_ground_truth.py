"""Extract ground truth positions from labelled images."""

import glob
import os
import re
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage.measurements import center_of_mass

import analysis.images as im


def main():

    load_dir = os.path.join('data', 'kinect', 'labelled_trials')

    labelled_trial_names = os.listdir(load_dir)

    body_part_grayscale = OrderedDict({
        'HEAD': 105,
        'L_HIP': 93,
        'R_HIP': 40,
        'L_THIGH': 133,
        'R_THIGH': 240,
        'L_KNEE': 52,
        'R_KNEE': 111,
        'L_CALF': 36,
        'R_CALF': 37,
        'L_FOOT': 135,
        'R_FOOT': 76,
    })

    # Camera calibration parameters
    x_res, y_res = 565, 430
    f_xz, f_yz = 1.11146664619446, 0.833599984645844

    # Regex to extract frame number from file name
    pattern = re.compile(r'(\d+)\.png')

    part_names, part_labels = zip(*body_part_grayscale.items())

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

            label_image = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

            if np.all(np.in1d(part_labels, np.unique(label_image))):
                # All body parts are visible in the label image

                centroids = np.array([
                    center_of_mass(label_image == label)
                    for label in part_labels
                ])

                # Round to nearest integer so that image can be indexed
                centroid_coords = np.round(centroids).astype(int)

                # Get depths and convert from mm to cm
                depths = depth_image[centroid_coords[:, 0],
                                     centroid_coords[:, 1]] / 10

                # Points in image space, in form (x, y, z)
                points_image = np.column_stack((np.fliplr(centroid_coords),
                                                depths))

                # Convert to real coordinates
                points_real = np.apply_along_axis(im.image_to_real, 1,
                                                  points_image, x_res, y_res,
                                                  f_xz, f_yz)

                for part_name, point in zip(part_names, points_real):
                    df_trial.loc[frame, part_name] = point

        df_trial.dropna(axis=0, inplace=True)

        dict_truth[trial_name] = df_trial

    # True positions from all labelled trials
    df_truth = pd.concat(dict_truth)
    df_truth.index.names = ['trial_name', 'frame']

    df_truth.to_pickle(os.path.join('results', 'dataframes', 'df_truth.pkl'))

if __name__ == '__main__':
    main()
