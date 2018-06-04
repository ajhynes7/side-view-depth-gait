import numpy as np
import pandas as pd

from numpy.linalg import norm

import modules.general as gen
import modules.linear_algebra as lin
import modules.clustering as cl
import modules.math_funcs as mf


def foot_dist_peaks(foot_dist, frame_labels):

    frames = foot_dist.index.values.reshape(-1, 1)

    # Upper foot distance values are those above
    # the root mean square value
    rms = mf.root_mean_square(foot_dist.values)
    is_upper_value = foot_dist > rms

    n_labels = frame_labels.max() + 1
    frame_list = []

    # Each label represent one walking pass by the camera
    for i in range(n_labels):

        # Upper foot distance values of one walking pass
        upper_of_pass = (frame_labels == i) & is_upper_value

        # Find centres of foot distance peaks with mean shift
        input_frames = frames[upper_of_pass]
        _, centroids, k = cl.mean_shift(input_frames,
                                        cl.gaussian_kernel_shift, radius=5)

        # Find the frames closest to the mean shift centroids
        upper_pass_frames = frames[upper_of_pass]
        centroid_frames = [lin.closest_point(upper_pass_frames,
                                             x)[0].item() for x in centroids]

        frame_list.append(centroid_frames)

    # Flatten list and sort to obtain peak frames from whole walking trial
    peak_frames = sorted([x for sublist in frame_list for x in sublist])

    return peak_frames


def get_gait_metrics(df, frame_i, frame_f):
    """
    Uses two consecutive peak frames to calculate gait metrics.
    The peak frames are from the foot-to-foot distance data.
    Two consecutive peaks indicate a full walking stride.

    Parameters
    ----------
    df : DataFrame
        | Index is the frame numbers
        | Columns include 'HEAD', 'L_FOOT', 'R_FOOT'
        | Each element is a position vector
    frame_i : int
        Initial peak frame
    frame_f : int
        Final peak frame

    Returns
    -------
    metrics : dict
        Gait metrics
    """
    Head_i, Head_f = df.loc[frame_i, 'HEAD'], df.loc[frame_f, 'HEAD']

    L_foot_i, R_foot_i = df.loc[frame_i, 'L_FOOT'], df.loc[frame_i, 'R_FOOT']
    L_foot_f, R_foot_f = df.loc[frame_f, 'L_FOOT'], df.loc[frame_f, 'R_FOOT']

    dist_L, dist_R = norm(L_foot_f - L_foot_i), norm(R_foot_f - R_foot_i)

    # The stance foot is the one that moved the smaller distance
    swing_num = np.argmax([dist_L, dist_R])
    stance_num = ~swing_num

    points_i, points_f = [L_foot_i, R_foot_i], [L_foot_f, R_foot_f]

    P_swing_i, P_swing_f = points_i[swing_num], points_f[swing_num]
    P_stance_i, P_stance_f = points_i[stance_num], points_f[stance_num]
    P_stance = np.mean(np.vstack((P_stance_f, P_stance_i)), axis=0)

    P_proj = lin.proj_point_line(P_stance, P_swing_i, P_swing_f)

    step_length_i = norm(P_proj - P_swing_i)
    step_length_f = norm(P_proj - P_swing_f)

    # Divide frame difference by 30, because frame rate is 30 fps
    stride_time = (frame_f - frame_i) / 30

    metrics = {'Stride length': norm(P_swing_f - P_swing_i),
               'Stride width':  norm(P_stance - P_proj),

               'Stride vel':    norm(Head_f - Head_i) / stride_time,

               'Step length':   np.mean((step_length_i, step_length_f))
               }

    return metrics


def gait_dataframe(df, peak_frames, label_dict):
    """
    Produces a pandas DataFrame containing gait metrics from a walking trial.

    Parameters
    ----------
    df : DataFrame
        | Index is the frame numbers
        | Columns include 'HEAD', 'L_FOOT', 'R_FOOT'
        | Each element is a position vector
    peak_frames : array_like
        Array of all frames with a detected peak in the foot distance data
    label_dict : dict
        | Label of each peak frame
        | The labels are determined by clustering the peak frames

    Returns
    -------
    gait_df : DataFrame
        | Index is final peak frame used to calculate gait metrics
        | Columns are gait metric names
    """
    gait_list, frame_list = [], []

    for frame_i, frame_f in gen.pairwise(peak_frames):

        if label_dict[frame_i] == label_dict[frame_f]:
            metrics = get_gait_metrics(df, frame_i, frame_f)

            gait_list.append(metrics)
            frame_list.append(frame_f)

    gait_df = pd.DataFrame(gait_list, index=frame_list)
    gait_df.index.name = 'Frame'

    return gait_df
