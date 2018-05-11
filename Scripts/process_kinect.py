import os
import pandas as pd
import numpy as np

file_name = '2014-12-22_P007_Pre_004.txt'
load_dir = '../../../MEGA/Data/Kinect Zeno/Kinect trials'
save_dir = '../../../MEGA/Data/Kinect Zeno/Kinect processed'

load_path = os.path.join(load_dir, file_name)

# Number of columns for the position coordinates
# Number should be sufficiently large and divisible by 3
n_coord_cols = 90

df = pd.read_csv(load_path, skiprows=range(22), header=None,\
    names = [i for i in range(-2, n_coord_cols)], sep='\t', engine='python')

# Change some column names
df.rename(columns={-2: 'Frame', -1: 'Part'}, inplace=True)

# Replace any non-number strings with nan in the Frame column
df['Frame'] = df['Frame'].replace(r'[^0-9]', np.nan, regex=True)

# Convert the strings in the frame column to numbers
df['Frame'] = pd.to_numeric(df['Frame'])

max_frame = max(df['Frame'])
n_frames = int(max_frame) + 1

# Crop the dataframe at the max frame number
last_index = df[df.Frame == max_frame].index[-1]
df = df.loc[:last_index, :]

df.set_index('Frame', inplace=True)

# Part names
parts = df.groupby('Part').groups.keys()

population_list = []

for part in parts:
    df_part = df[df.Part == part]
    
    population_dict = {}
    
    for frame in range(n_frames):
        
        part_vector = df_part.iloc[frame][1:].dropna()  
        
        # Reshape array into an n x 3 matrix. Each row is an x, y, z position
        # The -1 means the row dimension is inferred
        population = part_vector.values.reshape(-1, 3).astype(float)
        population.round(2) # Round to save space
        
        population_dict[frame] = population
        
    population_list.append(population_dict)


df_final = pd.DataFrame(population_list).T
df_final.columns = parts
df_final.index.name = 'Frame'

save_name = file_name.replace(".txt", "")
save_path = os.path.join(save_dir, save_name)
df_final.to_pickle(save_path)
    