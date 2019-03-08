"""Plot label and depth images with ground truth positions."""

import glob
import os
from os.path import join
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.images as im


def main():

    load_dir = join('data', 'kinect', 'labelled_trials')
    align_dir = join('data', 'kinect', 'alignment')

    labelled_trial_names = os.listdir(load_dir)

    trial_name = labelled_trial_names[0]
    label_dir = join(load_dir, trial_name, 'label')
    depth_dir = join(load_dir, trial_name, 'depth16bit')

    label_paths = sorted(glob.glob(join(label_dir, '*.png')))
    depth_paths = sorted(glob.glob(join(depth_dir, '*.png')))

    df_truth = pd.read_pickle(join('results', 'dataframes', 'df_truth.pkl'))

    image_number = 318
    label_path = [x for x in label_paths if str(image_number) in x][0]
    depth_path = [x for x in depth_paths if str(image_number) in x][0]

    label_image = cv2.imread(label_path, cv2.IMREAD_ANYCOLOR)
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

    # Load dictionary to convert image numbers to frames
    with open(join(align_dir, "{}.pkl".format(trial_name)), 'rb') as handle:
        image_to_frame = pickle.load(handle)

    frame = image_to_frame[image_number]

    points_real = np.stack(df_truth.loc[trial_name, frame])
    points_image = np.apply_along_axis(
        im.real_to_image, 1, points_real, im.X_RES, im.Y_RES, im.F_XZ, im.F_YZ
    )

    # Label image
    fig = plt.figure()

    plt.imshow(label_image)
    plt.scatter(points_image[:, 0], points_image[:, 1], c='w', edgecolor='k')
    plt.axis('off')

    fig.savefig(join('figures', 'label_image'))

    # Depth image
    fig = plt.figure()

    plt.scatter(points_image[:, 0], points_image[:, 1], c='w')
    plt.imshow(depth_image, cmap='gray')

    plt.scatter(points_image[:, 0], points_image[:, 1], c='w', edgecolor='k')
    plt.axis('off')

    fig.savefig(join('figures', 'depth_image'))


if __name__ == '__main__':
    main()
