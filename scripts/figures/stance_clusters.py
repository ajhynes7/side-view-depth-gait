"""Plot detected clusters representing stance phases."""

from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skspatial.transformation import transform_coordinates

import modules.numpy_funcs as nf
import modules.phase_detection as pde


def main():

    # %% Load data

    trial_name, num_pass = '2014-12-08_P007_Post_003', 0

    df_selected_passes = pd.read_pickle(join('data', 'kinect', 'df_selected_passes.pkl'))
    df_pass = df_selected_passes.loc[(trial_name, num_pass)]

    points_head = np.stack(df_pass.HEAD)
    points_a = np.stack(df_pass.L_FOOT)
    points_b = np.stack(df_pass.R_FOOT)

    # %% Cluster stance phases

    # Group points together by alternating from groups A and B.
    # This ensures that the points move in one general direction.
    points_foot = nf.interweave_rows(points_a, points_b)

    model_ransac, is_inlier = pde.fit_ransac(points_foot)
    basis = pde.compute_basis(points_head, points_a, points_b, model_ransac)

    # Keep only foot points marked as inliers by RANSAC.
    points_foot_inlier = points_foot[is_inlier]

    # Convert foot points into new coordinates defined by forward, up, and perpendicular directions.
    points_transformed = transform_coordinates(points_foot_inlier, basis.origin, (basis.up, basis.perp, basis.forward))
    coords_up, coords_perp, coords_forward = np.split(points_transformed, 3, 1)

    points_2d = np.column_stack((coords_perp, coords_forward))
    labels_stance = pde.label_stance_phases(points_2d)

    # %% Plot figure

    x, y = coords_perp.flatten(), coords_forward.flatten()
    is_noise = labels_stance == -1

    fig = plt.figure()

    plt.scatter(x[is_noise], y[is_noise], c='k', s=10)

    plt.scatter(x[~is_noise], y[~is_noise], c=labels_stance[~is_noise], cmap='Set1', edgecolor='k', s=50)

    plt.xlabel('Perpendicular coordinates [cm]')
    plt.ylabel('Forward coordinates [cm]')

    plt.axis([-50, 50, -160, 160])

    fig.savefig(join('figures', 'stance_clusters.pdf'), dpi=1200)


if __name__ == '__main__':
    main()
