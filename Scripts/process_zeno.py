import pandas as pd
import glob
import os

directory = '../../../MEGA/Data/Zeno trials/'
path_string = directory + '*.xlsx'

file_names = [os.path.basename(x) for x in glob.glob(path_string)]

names = ['Step Length (cm.)', 'Stride Length (cm.)',
         'Stride Width (cm.)', 'Stride Velocity (cm./sec.)']

gait_list = []

for f in file_names:

    file_path = os.path.join(directory, f)

    # Two Excel sheet formats were used when collecting the data,
    # so the number of skipped rows changes
    n_to_skip = 5 if f[0] == 'A' else 11

    df = pd.read_excel(file_path, skiprows=n_to_skip)

    # %% Extract relvant gait metrics from spreadsheet
    df = df.rename(columns={'Unnamed: 0': 'Measurement', 'Unnamed: 1': 'Type'})

    names = ['Step Length (cm.)', 'Stride Length (cm.)',
             'Stride Width (cm.)', 'Stride Velocity (cm./sec.)']

    is_mean_val = (df['Measurement'] == 'Mean') & (df['Type'].isnull())

    # Create dictionary from the slice of the dataframe
    gait_metrics = df[names][is_mean_val].to_dict('records')[0]
    gait_metrics['File'] = f

    gait_list.append(gait_metrics)

# The list of dicts is converted into a final dataframe
df_final = pd.DataFrame(gait_list)
