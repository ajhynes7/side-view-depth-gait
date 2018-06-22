import os
import glob
import copy

import pandas as pd
import numpy as np


def main():

    load_dir = os.path.join('..', 'data', 'kinect', 'raw')

    # Directories for all hypothetical points and highest confidence points
    save_dir_hypo = os.path.join('..', 'data', 'kinect', 'processed',
                                 'hypothesis')

    save_dir_conf = os.path.join('..', 'data', 'kinect', 'processed',
                                 'confidence')

    # All files with .txt extension
    file_paths = glob.glob(os.path.join(load_dir, '*.txt'))

    # Number of columns for the position coordinates
    # Number should be sufficiently large and divisible by 3
    n_coord_cols = 99

    for file_path in file_paths[10:30]:

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

        # Crop the DataFrame at the max frame number
        last_index = df[df.Frame == max_frame].index[-1]
        df = df.loc[:last_index, :]

        # Part names
        parts = df.groupby('Part').groups.keys()

        df_hypo = pd.concat([df.loc[:, ['Frame', 'Part']],
                             df.iloc[:, 5:]], axis=1)

        df_conf = df.iloc[:, range(5)]

        has_data = df_hypo.loc[:, 3].notnull()

        df_hypo, df_conf = df_hypo[has_data], df_conf[has_data]

        coord_counts = np.sum(df_hypo.iloc[:, 2:].notnull(), axis=1)

        dict_hypo = {part: {f: np.nan for f in range(max_frame + 1)}
                     for part in parts}
        dict_conf = copy.deepcopy(dict_hypo)

        df_iterator = zip(df_hypo.iterrows(), df_conf.iterrows(), coord_counts)

        for (index, row_hypo), (_, row_conf), n_coords in df_iterator:

            frame, part = row_hypo.Frame, row_hypo.Part

            coordinates = row_hypo.values[2: 2 + n_coords]
            points = coordinates.reshape(-1, 3)

            dict_hypo[part][frame] = points
            dict_conf[part][frame] = row_conf.values[2:]

        df_final_hypo = pd.DataFrame(dict_hypo).rename_axis('Frame')
        df_final_conf = pd.DataFrame(dict_conf).rename_axis('Frame')

        # Save data

        base_name = os.path.basename(file_path)     # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        save_path_hypo = os.path.join(save_dir_hypo, file_name) + '.pkl'
        save_path_conf = os.path.join(save_dir_conf, file_name) + '.pkl'

        df_final_hypo.to_pickle(save_path_hypo)
        df_final_conf.to_pickle(save_path_conf)


if __name__ == '__main__':

    main()
