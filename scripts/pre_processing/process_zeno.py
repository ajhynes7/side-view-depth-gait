"""Process data from Excel files with Zeno Walkway measurements."""

import glob
import re
from os.path import basename, join, splitext

import pandas as pd


def extract_measurements(df_raw):
    """Extract gait parameter measurements from the raw Zeno data."""
    row_param_names = df_raw.apply(lambda row: row.str.contains('Stride Length').any(), axis=1).idxmax()
    row_data_begins = (df_raw.iloc[:, 0] == 1).idxmax()

    df_trial = df_raw.iloc[row_data_begins:]

    df_trial.columns = df_raw.iloc[row_param_names]
    df_trial.columns.name = None

    # The first column is an irrelevant row count.
    return df_trial.iloc[:, 1:].reset_index(drop=True)


def parse_walking_info(df_trial):
    """Parse stride information (e.g. pass number, foot side)."""

    def yield_parsed(series_info):

        for string in series_info:

            first_char = string[0]

            if first_char.isdigit():
                # This row starts a new walking pass.
                # Update current pass number.
                # Subtract one to match zero-indexing of Kinect passes.
                num_pass = int(first_char) - 1

            # Match 'Right' or 'Left' and take first character ('R' or 'L')
            side = re.search(r'(\w+)\s', string).group(1)[0]

            yield num_pass, side

    series_info = df_trial.iloc[:, 0]
    df_parsed = pd.DataFrame(yield_parsed(series_info), columns=['num_pass', 'side'])

    return pd.concat((df_parsed, df_trial), axis=1).set_index(df_parsed.columns.to_list())


def select_parameters(df_trial):
    """Select gait parameters of interest and simplify the column names."""

    dict_labels = {
        'Absolute Step Length (cm.)': 'absolute_step_length',
        'Step Length (cm.)': 'step_length',
        'Stride Length (cm.)': 'stride_length',
        'Stride Width (cm.)': 'stride_width',
        'Stride Velocity (cm./sec.)': 'stride_velocity',
        'Stride Time (sec.)': 'stride_time',
        'Stance %': 'stance_percentage',
        'Toe In/Out Angle (degrees)': 'toe_angle',
        'Foot Length (cm.)': 'foot_length',
        'Foot Area (cm. x cm.)': 'foot_area',

    }

    return df_trial[dict_labels].rename(dict_labels, axis=1).dropna().astype(float)


def main():

    # All files with .xlsx extension
    load_dir = join('data', 'zeno', 'raw')
    file_paths = sorted(glob.glob(join(load_dir, '*.xlsx')))

    dict_trials = {}

    for file_path in file_paths:

        df_trial = pd.read_excel(file_path).pipe(extract_measurements).pipe(parse_walking_info).pipe(select_parameters)

        trial_name = splitext(basename(file_path))[0]
        dict_trials[trial_name] = df_trial

    df_gait = pd.concat(dict_trials).dropna()
    df_gait.index = df_gait.index.rename(level=0, names='trial_name')

    df_gait.to_pickle(join('data', 'zeno', 'df_gait.pkl'))


if __name__ == '__main__':
    main()
