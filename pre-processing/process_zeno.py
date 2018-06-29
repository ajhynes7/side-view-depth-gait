import glob
import os

import numpy as np
import pandas as pd


def main():

    load_dir = os.path.join('..', 'data', 'zeno', 'raw')
    save_dir = os.path.join('..', 'data', 'results')

    save_name = 'zeno_gait_metrics.csv'

    labels = ['Step Length (cm.)', 'Stride Length (cm.)',
              'Stride Width (cm.)', 'Stride Velocity (cm./sec.)',
              'Absolute Step Length (cm.)', 'Stride Time (sec.)']

    new_labels = ['step_length', 'stride_length',
                  'stride_width', 'stride_velocity',
                  'absolute_step_length', 'stride_time']

    label_dict = {k: v for k, v in zip(labels, new_labels)}

    # All files with .xlsx extension
    file_paths = glob.glob(os.path.join(load_dir, '*.xlsx'))

    save_path = os.path.join(save_dir, save_name)

    # %% Read gait metrics from each Zeno file

    list_l, list_r = [], []

    # Useful for catching errors with relative file paths
    assert len(file_paths) > 0

    for file_path in file_paths:

        base_name = os.path.basename(file_path)     # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        df = pd.read_excel(file_path)

        # Locate the gait metric labels in the Excel file
        bool_array = df.applymap(lambda x: 'Step Time' in x if
                                 isinstance(x, str) else False).values

        row_gait, col_gait = np.argwhere(bool_array)[0]
        df.columns = df.iloc[row_gait, :]

        df_gait = df.iloc[row_gait + 1:]

        df_gait.columns.values[0] = 'Type'
        df_gait.columns.values[1] = 'Side'
        df_gait = df_gait.set_index(['Type', 'Side'])

        series_l = df_gait.loc['Mean'].loc['Left'][labels]
        series_r = df_gait.loc['Mean'].loc['Right'][labels]

        series_l['File'], series_r['File'] = file_name, file_name

        list_l.append(series_l)
        list_r.append(series_r)

    df_l = pd.DataFrame(list_l)
    df_r = pd.DataFrame(list_r)

    # Change column labels to new labels
    df_l = df_l.rename(label_dict, axis='columns')
    df_r = df_r.rename(label_dict, axis='columns')

    # Merge left and right DataFrames by matching filenames
    df_final = pd.merge(df_l, df_r, left_on='File', right_on='File',
                        suffixes=('_L', '_R'))

    df_final = df_final.set_index('File')

    # Order columns alphabetically
    df_final = df_final.reindex(sorted(df_final.columns), axis=1)

    df_final.to_csv(save_path, index=True)


if __name__ == '__main__':

    main()
