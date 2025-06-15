import numpy as np
from observers.eso import ESO
from .controller import Controller
from numpy.linalg import inv

class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp, m3 = 0.0, r3 = 0.01):
        self.b = b
        self.kp = kp
        self.kd = kd
        self.last_u = 0

        self.cnt = 0

        A = np.eye(3,k=1)
        B = np.array([[0], [self.b], [0]])
        L = np.array([[3*p], [3*p**2], [p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)

        self.Tp = Tp
        self.l1 = 0.5
        self.r1 = 0.04
        self.m1 = 3.
        self.l2 = 0.4
        self.r2 = 0.04
        self.m2 = 2.4
        self.I_1 = 1. / 12 * self.m1 * (3 * self.r1 ** 2 + self.l1 ** 2)
        self.I_2 = 1. / 12 * self.m2 * (3 * self.r2 ** 2 + self.l2 ** 2)
        self.m3 = m3
        self.r3 = r3
        self.I_3 = 2. / 5 * self.m3 * self.r3 ** 2

        self.d1 = self.l1/2
        self.d2 = self.l2/2
        
    def M(self, x):
        """
        Please implement the calculation of the mass matrix, according to the model derived in the exercise
        (2DoF planar manipulator with the object at the tip)
        """
        q1, q2 = x
        
        alpha = (self.I_1 + self.I_2 + self.I_3 + self.m1 * pow(self.d1, 2) + self.m2 * (pow(self.l1, 2) + pow(self.d2, 2))
                 + self.m3 * (pow(self.l1, 2) + pow(self.l2, 2)) + 2 * self.m3 * self.l1 * self.l2)
        gamma = self.m2 * pow(self.d2, 2) + self.m3 * (pow(self.l2, 2)) + self.I_2 + self.I_3
        beta = self.m2*self.l1*self.d2 + self.m3*self.l1*self.l2
        

        m11 = alpha + 2*beta*np.cos(q2)
        m12 = gamma+beta*np.cos(q2)
        m21 = gamma+beta*np.cos(q2)
        m22 = gamma
        Mass = np.array([[m11 , m12],[m21, m22]])

        return Mass

    def C(self, x):
        """
        Please implement the calculation of the Coriolis and centrifugal forces matrix, according to the model derived
        in the exercise (2DoF planar manipulator with the object at the tip)
        """
        q1, q2, q1_dot, q2_dot = x

        beta = (self.m2*self.l1*self.d2) + (self.m3*self.l1*self.l2)
        

        c11 = -beta*np.sin(q2)*q2_dot
        c12 = -beta*np.sin(q2)*(q1_dot+q2_dot)
        c21 = beta*np.sin(q2)*q1_dot
        c22 = 0
        Coriolis = np.array([[c11 , c12],[c21, c22]])


        return Coriolis

    def x_dot(self, x, u):
            invM = np.linalg.inv(self.M(x))
            zeros = np.zeros((2, 2), dtype=np.float32)
            A = np.concatenate([np.concatenate([zeros, np.eye(2)], 1), np.concatenate([zeros, -invM @ self.C(x)], 1)], 0)
            b = np.concatenate([zeros, invM], 0)
            return A @ x[:, np.newaxis] + b @ u

    def set_b(self, b):
        self.eso.set_B(np.array([[0], [b], [0]]))

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q, q_dot = x
        q_est, q_est_dot, f = self.eso.get_state()
        self.eso.update(q, self.last_u)
        v = self.kp * (q_d - q) + self.kd * (q_d_dot - q_est_dot) + q_d_ddot
        u = (v - f) / self.b
        self.last_u = u
        return u