import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.models = [ManiuplatorModel(Tp,0.1,0.05), ManiuplatorModel(Tp,0.01,0.01), ManiuplatorModel(Tp,1.0,0.3)]
        self.i = 0
        self.Tp = Tp
        self.prev_x = np.zeros(4)
        self.prev_u = np.zeros(2)

    def choose_model(self, x):
        x_1, x_2, x_3 = [model.x_dot(self.prev_x, self.prev_u) * self.Tp + self.prev_x.reshape(4,1) for model in self.models]
        errors = [np.sum(abs(x.reshape(4,1) - x_1)), np.sum(abs(x.reshape(4,1) - x_2)), np.sum(abs(x.reshape(4,1) - x_3))]
        self.i = errors.index(min(errors))


    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        K_d = [[62, 0], [0, 62]]
        K_p = [[95, 0], [0, 95]]
        v = q_r_ddot + K_d @ (q_r_dot - q_dot) + K_p @ (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.prev_u, self.prev_x = u, x
        return u
