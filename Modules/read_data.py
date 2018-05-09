import numpy as np


def read_positions(frame_df, part, max_num_coords):
    """
    Inputs
    ------

    Outputs
    -------
    """
    row = frame_df[frame_df['Part'] == part]
    row_coordinates = row.loc[:, range(max_num_coords)].as_matrix()

    n_coordinates = np.sum(~np.isnan(row_coordinates))
    n_points = int(n_coordinates / 3)

    points = np.full([n_points, 3], np.nan)

    for idx, i in enumerate(range(0, n_coordinates, 3)):
        points[idx, :] = row_coordinates[0, i:i+3]

    return points
