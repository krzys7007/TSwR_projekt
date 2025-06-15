import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, x, q_d, q_dot, q_d_dot):
        q = [x[0], x[1]]
        q_dot = [x[2], x[3]]
        u = self.kp * (q_d - q) + self.kd * (q_d_dot - q_dot)
        return u
