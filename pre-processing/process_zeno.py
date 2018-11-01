"""Process data from Excel files with Zeno Walkway measurements."""

import glob
import os
import re

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

    return pd.DataFrame({
        'pass': pass_list,
        'stride': stride_list,
        'side': side_list,
    })


def main():

    # Useful for catching errors with relative file paths
    assert len(file_paths) > 0

    for file_path in file_paths:

        df = pd.read_excel(file_path)

        # Locate the gait metric labels in the Excel file
        bool_array = df.applymap(
            lambda x: 'Step Time' in x if isinstance(x, str) else False).values

        # Crop DataFrame at row where raw values begin
        row_first_data = int(np.where(df.iloc[:, 0] == 1)[0])
        df_gait = df.iloc[row_first_data:, :]

        # Set the gait parameters as column labels
        row_gait_parameters = np.argwhere(bool_array)[0][0]
        df_gait.columns = df.iloc[row_gait_parameters, :]

        df_gait.columns.name = None

        df_labels = df_gait[labels].rename(label_dict, axis=1)
        df_labels = df_labels.reset_index(drop=True)

        stride_info = df_gait.iloc[:, 1]
        df_numbers = parse_stride_info(stride_info)

        df_trial = pd.concat((df_numbers, df_labels), axis=1, sort=False)

        base_name = os.path.basename(file_path)  # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        save_path = os.path.join(save_dir, file_name + '.pkl')
        df_trial.to_pickle(save_path)


load_dir = os.path.join('data', 'zeno', 'raw')
save_dir = os.path.join('data', 'zeno', 'gait_params')

labels = [
    'Step Length (cm.)', 'Stride Length (cm.)', 'Stride Width (cm.)',
    'Stride Velocity (cm./sec.)', 'Stride Time (sec.)',
    'Stance %', 'Total D. Support %',
]

new_labels = [
    'step_length', 'stride_length', 'stride_width', 'stride_velocity',
    'stride_time', 'stance_percentage', 'stance_percentage_double'
]

label_dict = {k: v for k, v in zip(labels, new_labels)}

# All files with .xlsx extension
file_paths = sorted(glob.glob(os.path.join(load_dir, '*.xlsx')))


if __name__ == '__main__':

    main()
