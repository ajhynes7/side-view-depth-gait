"""Process data from Excel files with Zeno Walkway measurements."""

import glob
import re
from os.path import basename, join, splitext

import numpy as np
import pandas as pd


def parse_stride_info(stride_info):
    """Parse stride information (e.g. pass number, stride number)."""
    pass_list, side_list, stride_list = [], [], []

    for string in stride_info:

        first_char = string[0]

        if first_char.isdigit():
            # This row starts a new walking pass
            # Update current pass number
            pass_number = int(first_char)

        pass_list.append(pass_number)

        # Match 'Right' or 'Left' and take first character ('R' or 'L')
        side_list.append(re.search(r'(\w+)\s', string).group(1)[0])

        stride_list.append(int(string[-1]))

    return pd.DataFrame({'pass': pass_list, 'stride': stride_list, 'side': side_list})


def main():

    labels = [
        'Absolute Step Length (cm.)',
        'Step Length (cm.)',
        'Stride Length (cm.)',
        'Stride Width (cm.)',
        'Stride Velocity (cm./sec.)',
        'Stride Time (sec.)',
        'Stance %',
    ]

    new_labels = [
        'absolute_step_length',
        'step_length',
        'stride_length',
        'stride_width',
        'stride_velocity',
        'stride_time',
        'stance_percentage',
    ]

    label_dict = {k: v for k, v in zip(labels, new_labels)}

    # All files with .xlsx extension
    load_dir = join('data', 'zeno', 'raw')
    file_paths = sorted(glob.glob(join(load_dir, '*.xlsx')))

    dict_trials = {}

    for file_path in file_paths:

        df = pd.read_excel(file_path)

        # Locate the gait parameter labels in the Excel file
        bool_array = df.applymap(lambda x: 'Step Time' in x if isinstance(x, str) else False).values

        # Crop DataFrame at row where raw values begin
        row_first_data = int(np.where(df.iloc[:, 0] == 1)[0])
        df_gait = df.iloc[row_first_data:, :]

        # Set the gait parameters as column labels
        row_gait_parameters = np.argwhere(bool_array)[0][0]
        df_gait.columns = df.iloc[row_gait_parameters, :]

        df_gait.columns.name = None

        df_labels = df_gait[labels].rename(label_dict, axis=1)
        df_labels = df_labels.reset_index(drop=True).astype(np.float)

        stride_info = df_gait.iloc[:, 1]
        df_numbers = parse_stride_info(stride_info)

        df_trial = pd.concat((df_numbers, df_labels), axis=1, sort=False)

        base_name = basename(file_path)  # File with extension
        file_name = splitext(base_name)[0]  # File with no extension

        dict_trials[file_name] = df_trial

    # DataFrame containing all Zeno trials
    df_gait = pd.concat(dict_trials).dropna()

    df_gait.to_pickle(join('data', 'zeno', 'df_gait.pkl'))


if __name__ == '__main__':
    main()
