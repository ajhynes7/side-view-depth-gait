from numpy.linalg import norm

import modules.linear_algebra as lin


class Stride():

    def __init__(self, state_i, state_f):

        self.stance_i = state_i.stance
        self.stance_f = state_f.stance

        self.swing_i = state_i.swing
        self.swing_f = state_f.swing

        self.frame_i = state_i.frame
        self.frame_f = state_f.frame

        self.stance = (self.stance_i + self.stance_f) / 2

        self.proj_stance = lin.proj_point_line(self.stance, self.swing_i,
                                               self.swing_f)

    @property
    def stride_length(self):
        return norm(self.swing_f - self.swing_i)

    @property
    def step_length(self):

        return norm(self.proj_stance - self.swing_i)

    @property
    def stride_width(self):
        return norm(self.proj_stance - self.stance)

    @property
    def absolute_step_length(self):

        return norm(self.stance - self.swing_i)

    @property
    def stride_time(self):

        return (self.frame_f - self.frame_i) / 30

    @property
    def stride_velocity(self):

        return self.stride_length / self.stride_time