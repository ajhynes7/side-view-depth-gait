"""Plot label and depth images with body segment centroids."""

import glob
import os

import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.images as im


def main():

    load_dir = os.path.join('data', 'kinect', 'labelled_trials')
    save_dir = os.path.join('results', 'plots')

    labelled_trial_names = os.listdir(load_dir)

    trial_name = labelled_trial_names[0]
    label_dir = os.path.join(load_dir, trial_name, 'label')
    depth_dir = os.path.join(load_dir, trial_name, 'depth16bit')

    label_paths = sorted(glob.glob(os.path.join(label_dir, '*.png')))
    depth_paths = sorted(glob.glob(os.path.join(depth_dir, '*.png')))

    # Camera calibration parameters
    x_res, y_res = 565, 430
    f_xz, f_yz = 1.11146664619446, 0.833599984645844

    df_truth = pd.read_pickle(
        os.path.join('results', 'dataframes', 'df_truth.pkl'))

    frame = 318
    label_path = [x for x in label_paths if str(frame) in x][0]
    depth_path = [x for x in depth_paths if str(frame) in x][0]

    label_image = cv2.imread(label_path, cv2.IMREAD_ANYCOLOR)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    points_real = np.stack(df_truth.loc[trial_name, frame])
    points_image = np.apply_along_axis(im.real_to_image, 1, points_real, x_res,
                                       y_res, f_xz, f_yz)

    # Label image
    fig = plt.figure()

    plt.imshow(label_image)
    plt.scatter(points_image[:, 0], points_image[:, 1], c='w', edgecolor='k')
    plt.axis('off')

    fig.savefig(os.path.join(save_dir, 'label_image'))

    # Depth image
    fig = plt.figure()

    plt.scatter(points_image[:, 0], points_image[:, 1], c='w')
    plt.imshow(depth_image)

    plt.scatter(points_image[:, 0], points_image[:, 1], c='w', edgecolor='k')
    plt.axis('off')

    fig.savefig(os.path.join(save_dir, 'depth_image'))


if __name__ == '__main__':
    main()
