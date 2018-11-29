"""Plot label and depth images with body segment centroids."""

import glob
import os
from os.path import join
import re

import cv2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analysis.images as im
import analysis.plotting as pl
import modules.pose_estimation as pe


def main():

    load_dir = join('data', 'kinect', 'labelled_trials')

    hypo_dir = join('data', 'kinect', 'processed', 'hypothesis')
    align_dir = join('data', 'kinect', 'alignment')

    part_types = ['Head', 'Hip', 'Thigh', 'Knee', 'Calf', 'Foot']
    file_index = 271

    labelled_trial_names = os.listdir(load_dir)
    trial_name = labelled_trial_names[0]

    depth_dir = join(load_dir, trial_name, 'depth16bit')
    depth_paths = sorted(glob.glob(join(depth_dir, '*.png')))

    depth_path = depth_paths[file_index]
    depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    df_hypo = pd.read_pickle(join(hypo_dir, trial_name) + '.pkl')

    # %% Convert image file numbers to frame numbers

    df_align = pd.read_csv(
        join(align_dir, trial_name + '.txt'),
        header=None,
        names=['image_file'])

    # Extract number from image file name
    pattern = r'(\d+)\.png'
    df_align['image_number'] = df_align.image_file.str.extract(pattern)
    df_align = df_align.dropna()
    df_align.image_number = pd.to_numeric(df_align.image_number)

    # Dictionary mapping image file numbers to frames
    image_to_frame = {
        image_num: frame
        for frame, image_num in enumerate(df_align.image_number.values)
    }

    match_object = re.search(r'(\d+).png', depth_path)
    image_number = int(match_object.group(1))
    frame = image_to_frame[image_number]

    hypotheses = df_hypo.loc[frame]

    part_labels = range(len(part_types))
    points_real, labels = pe.get_population(hypotheses, part_labels)

    points_image = np.apply_along_axis(im.real_to_image, 1, points_real,
                                       im.X_RES, im.Y_RES, im.F_XZ, im.F_YZ)

    # %% Plot depth image and joint proposals

    fig = plt.figure()

    plt.imshow(depth_image, cmap='gray')
    pl.scatter_labels(points_image[:, :2], labels, edgecolors='k', s=100)

    plt.legend(part_types, framealpha=1, loc='upper left', fontsize=12)
    plt.axis('off')

    save_path = join('figures', 'joint_proposals.pdf')
    fig.savefig(save_path, format='pdf')


if __name__ == '__main__':
    main()
