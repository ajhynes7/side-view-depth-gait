import numpy as np
import pandas as pd
from numpy.linalg import norm
from linear_algebra import dist_point_line


def get_gait_metrics(df, frame_i, frame_f):

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
    P_stance = np.mean(np.vstack((P_stance_f, P_stance_i)))
    
    P_proj = dist_point_line(P_stance, P_swing_i, P_swing_f)
    
    # Multiply frame difference by 30, because frame rate is 30 fps
    stride_time = 30 * (frame_f - frame_i)
    
    metrics = {'Stride length': norm(P_swing_f - P_swing_i),
               'Stride width' : norm(P_stance - P_proj),
               
               'Stride vel':    norm(Head_f - Head_i) / stride_time,
               
               'Step length i': norm(P_proj - P_swing_i),
               'Step length f': norm(P_proj - P_swing_f)
               }
    
    return metrics
            

def gait_dataframe(df, peak_frames, frame_labels):
    
    gait_list, frame_list = [], []
    
    n_peaks = peak_frames.size
    
    for i in range(1, n_peaks):
        
        if frame_labels[i] == frame_labels[i-1]:
            
            frame_i, frame_f = peak_frames[i-1], peak_frames[i]
            
            metrics = get_gait_metrics(df, frame_i, frame_f)
            
            gait_list.append(metrics)
            frame_list.append(frame_f)
    
    gait_df = pd.DataFrame(gait_list, index=frame_list)
    gait_df.index.name = 'Frame'
    
    return gait_df