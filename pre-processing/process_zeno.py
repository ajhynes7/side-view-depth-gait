import glob
import os

import pandas as pd


def main():

    gait_list = []

    for file_path in file_paths:

        base_name = os.path.basename(file_path)     # File with extension
        file_name = os.path.splitext(base_name)[0]  # File with no extension

        # Two Excel sheet formats were used when collecting the data,
        # so the number of skipped rows changes
        n_to_skip = 5 if file_name[0] == 'A' else 11

        df = pd.read_excel(file_path, skiprows=n_to_skip)

        # Extract relevant gait metrics from spreadsheet
        df = df.rename(columns={'Unnamed: 0': 'Measurement',
                                'Unnamed: 1': 'Type'})

        is_mean_val = (df.Measurement == 'Mean') & (df.Type.isnull())

        # Create dictionary from the slice of the DataFrame
        gait_dict = df[column_names][is_mean_val].to_dict('records')[0]
        gait_dict['File'] = file_name

        gait_list.append(gait_dict)

    # The list of dicts is converted into a final DataFrame
    df_final = pd.DataFrame(gait_list)

    # Rename the columns so they match the Kinect columns
    df_final = df_final.rename(name_dict, axis='columns')

    df_final.to_csv(save_path, index=False)


if __name__ == '__main__':

    load_dir = os.path.join('..', 'data', 'zeno', 'raw')
    save_dir = os.path.join('..', 'data', 'results')

    save_name = 'zeno_gait_metrics.csv'

    column_names = ['Step Length (cm.)', 'Stride Length (cm.)',
                    'Stride Width (cm.)', 'Stride Velocity (cm./sec.)']

    new_names = ['Step length', 'Stride length', 'Stride width', 'Stride vel']

    name_dict = {k: v for k, v in zip(column_names, new_names)}

    # All files with .xlsx extension
    file_paths = glob.glob(os.path.join(load_dir, '*.xlsx'))

    save_path = os.path.join(save_dir, save_name)

    main()
