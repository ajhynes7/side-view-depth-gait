"""
Transform raw data from the Kinect into a pandas DataFrame.

This script is intended to be run when the parent folder is the working
directory.

"""

import copy
import os

import numpy as np
import pandas as pd

import analysis.images as im


load_dir = os.path.join('data', 'kinect', 'raw')

# Directories for all hypothetical points and highest confidence points
save_dir_hypo = os.path.join('data', 'kinect', 'processed', 'hypothesis')
save_dir_conf = os.path.join('data', 'kinect', 'processed', 'confidence')

# Find the Kinect files that have matching Zeno files
match_dir = os.path.join('data', 'matching')
df_match = pd.read_csv(os.path.join(match_dir, 'match_kinect_zeno.csv'))

# Number of columns for the position coordinates
# Number should be sufficiently large and divisible by 3
n_coord_cols = 99

# Parameters for recalibrating positions
x_res_old, y_res_old = 640, 480
x_res, y_res = 565, 430

f_xz, f_yz = 1.11146664619446, 0.833599984645844

for file_name in df_match.kinect:

    file_path = os.path.join(load_dir, file_name + '.txt')

    df = pd.read_csv(
        file_path,
        skiprows=range(22),
        header=None,
        names=[i for i in range(-2, n_coord_cols)],
        sep='\t',
        engine='python')

    # Change some column names
    df.rename(columns={-2: 'Frame', -1: 'Part'}, inplace=True)

    # Replace any non-number strings with nan in the Frame column
    df.Frame = df.Frame.replace(r'[^0-9]', np.nan, regex=True)

    # Convert the strings in the frame column so the max function will work
    df.Frame = pd.to_numeric(df.Frame)

    max_frame = int(np.nanmax(df.Frame))

    # Crop the DataFrame at the max frame number
    last_index = df[df.Frame == max_frame].index[-1]
    df = df.loc[:last_index, :]

    # Part names
    parts = df.groupby('Part').groups.keys()

    df_hypo = pd.concat([df.loc[:, ['Frame', 'Part']], df.iloc[:, 5:]], axis=1)

    df_conf = df.iloc[:, range(5)]

    has_data = df_hypo.loc[:, 3].notnull()
    df_hypo, df_conf = df_hypo[has_data], df_conf[has_data]

    coord_counts = np.sum(df_hypo.iloc[:, 2:].notnull(), axis=1)

    dict_hypo = {
        part: {f: np.nan
               for f in range(max_frame + 1)}
        for part in parts
    }

    dict_conf = copy.deepcopy(dict_hypo)

    df_iterator = zip(df_hypo.iterrows(), df_conf.iterrows(), coord_counts)

    for (_, row_hypo), (_, row_conf), n_coords in df_iterator:

        frame, part = row_hypo.Frame, row_hypo.Part

        coords_hypo = row_hypo.values[2:2 + n_coords].astype(float)
        coords_conf = row_conf.values[2:].astype(float)

        points_hypo_old = coords_hypo.reshape(-1, 3)
        points_conf_old = coords_conf.reshape(-1, 3)

        # The hypothetical positions need to be converted from
        # real to image then back to real using new parameters.
        points_hypo = im.recalibrate_positions(
            points_hypo_old, x_res_old, y_res_old, x_res, y_res, f_xz, f_yz)

        # The confidence points must be converted from
        # image to real coordinates
        points_conf = np.apply_along_axis(im.image_to_real, 1, points_conf_old,
                                          x_res, y_res, f_xz, f_yz)

        dict_hypo[part][frame] = points_hypo
        dict_conf[part][frame] = points_conf

    df_final_hypo = pd.DataFrame(dict_hypo).rename_axis('Frame')
    df_final_conf = pd.DataFrame(dict_conf).rename_axis('Frame')

    # Save data

    base_name = os.path.basename(file_path)  # File with extension
    file_name = os.path.splitext(base_name)[0]  # File with no extension

    save_path_hypo = os.path.join(save_dir_hypo, file_name) + '.pkl'
    save_path_conf = os.path.join(save_dir_conf, file_name) + '.pkl'

    df_final_hypo.to_pickle(save_path_hypo)
    df_final_conf.to_pickle(save_path_conf)
