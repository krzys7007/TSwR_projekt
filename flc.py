import matplotlib.pyplot as plt
import numpy as np

from controllers.dummy_controller import DummyController
from controllers.feedback_linearization_controller import FeedbackLinearizationController
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.01
start = 0
end = 3

"""
Switch to FeedbackLinearizationController as soon as you implement it
"""
controller = FeedbackLinearizationController(Tp)
#controller = DummyController(Tp)

"""
Here you have some trajectory generators. You can use them to check your implementations.
At the end implement Point2point trajectory generator to move your manipulator to some desired state.
"""
# traj_gen = ConstantTorque(np.array([0., 1.0])[:, np.newaxis])
# traj_gen = Sinusoidal(np.array([0.2, 1.]), np.array([2., 2.]), np.array([0., 0.]))
traj_gen = Poly3(np.array([0., 0.]), np.array([np.pi/4, np.pi/6]), end)


Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)
timesteps = np.linspace(0., end, int(end / Tp))
Q_des_0 = []
Q_des_1 = []
Q_dot_des_0 = []
Q_dot_des_1 = []
Q_ddot_des_0 = []
Q_ddot_des_1 = []
for t in timesteps:
    q_d, q_d_dot, q_d_ddot = traj_gen.generate(t)
    Q_des_0.append(q_d[0])
    Q_des_1.append(q_d[1])
    Q_dot_des_0.append(q_d_dot[0])
    Q_dot_des_1.append(q_d_dot[1])
    Q_ddot_des_0.append(q_d_ddot[0])
    Q_ddot_des_1.append(q_d_ddot[1])
print(Q_dot_des_0)

test1 = np.diff(Q_dot_des_1)
test1 = test1 / Tp

"""
You can add here some plots of the state 'Q' (consists of q and q_dot), controls 'u', desired trajectory 'Q_d'
with respect to time 'T' to analyze what is going on in the system
"""

plt.subplot(221)
plt.plot(T, Q[:, 0], 'r')
plt.plot(T, Q_d[:, 0], 'b')
plt.subplot(222)
plt.plot(T, Q[:, 1], 'r')
plt.plot(T, Q_d[:, 1], 'b')
plt.subplot(223)
plt.plot(T, u[:, 0], 'r')
plt.plot(T, u[:, 1], 'b')
plt.show()
"""
plt.subplot(221)
plt.plot(T, Q_des_0, 'r')
plt.plot(T, Q_des_1, 'b')
plt.subplot(222)
plt.plot(T, Q_dot_des_0, 'r')
plt.plot(T, Q_dot_des_1, 'b')
plt.subplot(223)
plt.plot(T, Q_ddot_des_0, 'r')
plt.plot(T, Q_ddot_des_1, 'b')
plt.subplot(224)
plt.plot(test1, 'r')
plt.plot(Q_ddot_des_1, 'b')
plt.show()
"""


