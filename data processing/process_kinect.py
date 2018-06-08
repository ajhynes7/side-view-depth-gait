import os
import glob
import copy

import pandas as pd
import numpy as np


file_paths = glob.glob('../../MEGA/Data/Kinect Zeno/Kinect trials/*.txt')

load_dir = '../../MEGA/Data/Kinect Zeno/Kinect trials'
save_dir = '../../MEGA/Data/Kinect Zeno/Kinect processed'


# Number of columns for the position coordinates
# Number should be sufficiently large and divisible by 3
n_coord_cols = 90


for file_path in file_paths[-1:]:

    df = pd.read_csv(file_path, skiprows=range(22), header=None,
                     names=[i for i in range(-2, n_coord_cols)],
                     sep='\t', engine='python')

    # Change some column names
    df.rename(columns={-2: 'Frame', -1: 'Part'}, inplace=True)

    # Replace any non-number strings with nan in the Frame column
    df.Frame = df.Frame.replace(r'[^0-9]', np.nan, regex=True)

    # Convert the strings in the frame column to numbers
    df.Frame = pd.to_numeric(df.Frame)

    max_frame = int(max(df.Frame))

    # Crop the dataframe at the max frame number
    last_index = df[df.Frame == max_frame].index[-1]
    df = df.loc[:last_index, :]

    # Part names
    parts = df.groupby('Part').groups.keys()

    df_conf = df.iloc[:, range(5)]
    df_hypo = pd.concat([df.loc[:, ['Frame', 'Part']], df.iloc[:, 5:]], axis=1)

    has_data = df_hypo.loc[:, 3].notnull()
    df_hypo = df_hypo[has_data]

    point_counts = np.sum(df_hypo.iloc[:, 2:].notnull(), axis=1)

    dict_hypo = {part: {f: np.nan for f in range(max_frame + 1)}
                 for part in parts}
    dict_conf = copy.deepcopy(dict_hypo)

    for index, n_coords in point_counts.items():

        row_hypo = df_hypo.loc[index]
        row_conf = df_conf.loc[index]

        part = row_hypo.Part
        frame = row_hypo.Frame

        coordinates = row_hypo.iloc[2: 2 + n_coords].values
        points = coordinates.reshape(-1, 3)

        dict_hypo[part][frame] = points
        dict_conf[part][frame] = row_conf[2:].values

    df_hypo_final = pd.DataFrame(dict_hypo).rename_axis('Frame')
    df_conf_final = pd.DataFrame(dict_conf).rename_axis('Frame')

    # %%  Save data to pickles

    save_path_conf = file_path.replace(".txt", "_conf.pkl")
    df_conf_final.to_pickle(save_path_conf)

    save_path_hypo = file_path.replace(".txt", ".pkl")
    df_hypo_final.to_pickle(save_path_hypo)
