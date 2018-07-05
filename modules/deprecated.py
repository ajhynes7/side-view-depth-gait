import numpy as np
from numpy.linalg import norm


class HeadMetrics:

    def __init__(self, head_points, frames):

        self.head_i, self.head_f = head_points

        self.frame_i, self.frame_f = frames

    def __str__(self):

        string = "HeadMetrics(frame_i={self.frame_i}, frame_f={self.frame_f})"

        return string.format(self=self)

    @property
    def head_distance(self):

        return norm(self.head_f - self.head_i)

    @property
    def stride_time(self):

        return (self.frame_f - self.frame_i) / 30

    @property
    def stride_velocity(self):

        return self.head_distance / self.stride_time


class FootMetrics:

    def __init__(self, stance_feet, swing_feet, frames):

        self.stance_i, self.stance_f = stance_feet
        self.swing_i, self.swing_f = swing_feet

        self.frame_i, self.frame_f = frames

        self.stance = (self.stance_i + self.stance_f) / 2

        self.stance_proj = lin.proj_point_line(self.stance, self.swing_i,
                                               self.swing_f)

    def __str__(self):

        string = "FootMetrics(frame_i={self.frame_i}, frame_f={self.frame_f})"

        return string.format(self=self)

    @property
    def stride_length(self):

        return norm(self.swing_f - self.swing_i)

    @property
    def step_length(self):

        return norm(self.stance_proj - self.swing_i)

    @property
    def stride_width(self):

        return norm(self.stance_proj - self.stance)

    @property
    def absolute_step_length(self):

        return norm(self.stance - self.swing_i)


class Stride:

    def __init__(self, df, frames, side):

        self.frame_i, self.frame_f = frames

        self.foot_i = df.loc[self.frame_i, side]
        self.foot_f = df.loc[self.frame_f, side]

        self.side = side

    def __str__(self):

        string = "Stride(frame_i={self.frame_i}, frame_f={self.frame_f}, " \
                 "side={self.side})"

        return string.format(self=self)

    @property
    def stride_length(self):

        return norm(self.foot_f - self.foot_i)

    @property
    def stride_time(self):

        return (self.frame_f - self.frame_i) / 30

    @property
    def stride_velocity(self):

        return self.stride_length / self.stride_time


def head_metrics(df, frame_i, frame_f):

    head_points = df.HEAD[[frame_i, frame_f]]

    frames = frame_i, frame_f

    head_obj = HeadMetrics(head_points, frames)

    return gen.get_properties(HeadMetrics, head_obj)


def foot_metrics(df, frame_i, frame_f):

    foot_points_i = np.stack(df.loc[frame_i, ['L_FOOT', 'R_FOOT']])
    foot_points_f = np.stack(df.loc[frame_f, ['L_FOOT', 'R_FOOT']])

    points_i, points_f = assign_swing_stance(foot_points_i, foot_points_f)

    stance_i, swing_i = points_i
    stance_f, swing_f = points_f

    stance_feet = stance_i, stance_f
    swing_feet = swing_i, swing_f
    frames = frame_i, frame_f

    foot_obj = FootMetrics(stance_feet, swing_feet, frames)

    return gen.get_properties(FootMetrics, foot_obj)


def assign_swing_stance(foot_points_i, foot_points_f):
    """
    Assign initial and final foot points to the stance and swing foot.

    Returns the combination that minimizes
    the distance travelled by the stance foot.

    Parameters
    ----------
    foot_points_i : ndarray
        The two initial foot points (at start of stride).
    foot_points_f : ndarray
        The two final foot points (at end of stride).

    Returns
    -------
    points_i : ndarray
        Initial foot points in order of (stance, swing).
    points_f : ndarray
        Final foot points in order of (stance, swing).

    """
    min_dist = np.inf
    points_i, points_f = [], []

    for a in range(2):
        for b in range(2):

            stance_i = foot_points_i[a, :]
            stance_f = foot_points_f[b, :]

            swing_i = foot_points_i[1 - a, :]
            swing_f = foot_points_f[1 - b, :]

            d_stance = norm(stance_f - stance_i)

            if d_stance < min_dist:

                min_dist = d_stance

                points_i = np.array([stance_i, swing_i])
                points_f = np.array([stance_f, swing_f])

    return points_i, points_f


@staticmethod
def is_stride(stance, swing_i, swing_f):

    swing_same_side = swing_i.side == swing_f.side

    swing_consec = swing_i.contact_number == swing_f.contact_number - 1

    same_contact = swing_i.contact_number == stance.contact_number

    swing_stance_diff_side = swing_i.side != stance.side

    tests = [swing_same_side, swing_consec, same_contact,
                swing_stance_diff_side]

    return np.all(tests)


def is_stride(foot_a0, foot_b0, foot_a1):

    a_a_same_side = foot_a0.side == foot_a1.side

    a_a_consecutive_contacts = foot_a0.contact_number == foot_a1.contact_number - 1

    a_b_same_contacts = foot_a0.contact_number == foot_b0.contact_number

    # Verify that foot A and B are on different sides.
    a_b_different_side = foot_a0.side != foot_b0.side

    tests = [a_a_same_side, a_a_consecutive_contacts,
             a_b_same_contacts, a_b_different_side]

    return np.all(tests)


def is_step(foot_a0, foot_b0, foot_a1):

    # Verify that the A feet compose a stride
    a_is_stride = is_stride(foot_a0, foot_a1)

    # Verify that foot A and B are on different sides.
    a_b_different = foot_a0.side != foot_b0.side

    return a_is_stride and a_b_different


def foot_signal(foot_series, *, r_window=5):
    """
    Return a signal from foot data.

    Uses a sliding window of frames to compute a clean signal.
    The signal can be used to detect stance and swing phases of a walk.

    Parameters
    ----------
    foot_series : Series
        Index values are frames.
        Values are foot positions.
    r_window : int, optional
        Radius of sliding window (a number of frames).

    Returns
    -------
    signal : Series
        Index values are frames.
        Values are the foot signal.

    """
    frames = foot_series.index.values
    signal = pd.Series(index=frames)

    x_coords = pd.Series(np.stack(foot_series)[:, 0], index=frames)

    for f in frames:

        x_prev = x_coords.reindex(np.arange(f - r_window, f)).dropna()
        x_next = x_coords.reindex(np.arange(f, f + r_window)).dropna()

        if x_prev.empty or x_next.empty:
            continue

        result_prev = linregress(x_prev.index.values, x_prev.values)
        result_next = linregress(x_next.index.values, x_next.values)

        signal[f] = result_prev.slope - result_next.slope

    return signal