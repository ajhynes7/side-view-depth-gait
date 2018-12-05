"""Align labelled image numbers to the frame numbers."""

import os
from os.path import join
import pickle

import pandas as pd


def main():

    load_dir = join('data', 'kinect', 'labelled_trials')
    align_dir = join('data', 'kinect', 'alignment')

    labelled_trial_names = os.listdir(load_dir)

    # %% Convert image file numbers to frame numbers

    for trial_name in labelled_trial_names:

        df_align = pd.read_csv(
            os.path.join(align_dir, trial_name + '.txt'),
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

        save_path = join(align_dir, "{}.pkl".format(trial_name))
        with open(save_path, 'wb') as handle:
            pickle.dump(image_to_frame, handle)


if __name__ == '__main__':
    main()
