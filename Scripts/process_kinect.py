import os
import pandas as pd
import numpy as np

# Number of columns for the position coordinates
# Number should be sufficiently large and divisible by 3
n_coord_cols = 90

directory = '../../../MEGA/Data/Kinect Zeno/Kinect trials'
file_name = '2014-12-22_P007_Pre_004.txt'

path = os.path.join(directory, file_name)

df = pd.read_csv(path, skiprows=range(22), header=None,\
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

parts = df.groupby('Part').groups.keys() # Part names


# %% Convert to new dataframe

@profile
def foo():
    population_list = []
    
    for frame in range(n_frames):
    
        population_dict = {'Frame': frame}
        df_frame = df.loc[frame, :]
    
        for i, part in enumerate(parts):
    
            # Convert the dataframe into an array of coordinates
            # Begin at 1 to skip the part name
            part_vector = df_frame.iloc[0, 1:].as_matrix()
    
            # Reshape array into an n x 3 matrix. Each row is an x, y, z position
            # The -1 means the row dimension is inferred
            population = np.reshape(part_vector, (-1, 3))
    
            population_dict[part] = population
    
        population_list.append(population_dict)
    
    df_final = pd.DataFrame(population_list)
    df_final.set_index('Frame', inplace=True)

    return df_final

df_final = foo()

# %% Save new dataframe

save_name = file_name.replace(".txt", "")
df_final.to_pickle(save_name)


