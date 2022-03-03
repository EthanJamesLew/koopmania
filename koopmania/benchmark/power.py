import koopmania as km
from koopmania.system import ContinuousSystem
import numpy as np
from numpy import sqrt, arctan, sin, cos
atan = arctan


class PowerSystem(ContinuousSystem):
    def __init__(self):
        pass

    def gradient(self, time: float, x: np.ndarray) -> np.ndarray:
        R_s = 10e-4
        R_fd = 0.0214
        R_kd = 2.
        R_kq = 2.
        L_ls = 1.0791 * 10e-6
        L_md = 5. * 10e-5
        L_mq = 5.956 * 10e-6
        L_lfd = 5.45 * 10e-6
        L_lkd = 3.87 * 10e-6
        L_lkq = 1.87 * 10e-6
        omega_r = 6283.2
        K_p = 0.1
        K_i = 15.0
        C_1 = 1. * 10e-5
        L_1 = 0.1 * 10e-4
        R_1 = 5. * 10e-4
        C_2 = 1. * 10e-4
        R_2 = 145.8
        L_l = 2. * 10e-6
        R_l = 2. * 10e-5
        C_f1 = 0.1 * 10e-4
        L_CP1 = 1. * 10e-7
        R_CP1 = 1. * 10e-4
        C_f2 = 5. * 10e-4
        v_ref = 270
        tau = 1e-4

        wbaseline = np.array([
            168.09,
            113.9,
            -1.08 * 10e-10,
            169.4,
            -1.055 * 10e-10,
            270.9,
            187.2,
            270.0,
            185.3,
            269.9,
            185.3,
            269.8,
            3.6212,
            0.2414
        ])[:, np.newaxis]

        Omega = omega_r * np.array([
            [0., - (L_ls + L_md), 0., L_md, L_md],
            [L_ls + L_mq, 0, -L_mq, 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
        ])

        R_hat = np.diag([-R_s, -R_s, R_kq, R_fd, R_kd])

        L = np.array([
            [-(L_ls + L_mq), 0., L_mq, 0., 0.],
            [0., -(L_ls + L_md), 0., L_md, L_md],
            [-L_mq, 0., L_lkq + L_mq, 0., 0.],
            [0., -L_md, 0., L_lfd + L_md, L_md],
            [0., -L_md, 0., L_md, L_lkd + L_md]
        ])

        alhpa = 0.6674
        beta = 0.9219
        phi = 0.1775
        i_q = x[1 - 1]
        i_d = x[2 - 1]
        i_kq = x[3 - 1]
        i_f = x[4 - 1]
        i_kd = x[5 - 1]
        v_dc = x[6 - 1]
        i_L1 = x[7 - 1]
        v_cgen = x[8 - 1]
        i_load = x[9 - 1]
        v_Cf1 = x[10 - 1]
        i_CP1 = x[11 - 1]
        v_Cf2 = x[12 - 1]
        v_f = x[13 - 1]
        int_tracking_err = x[14 - 1]

        dxdt = np.zeros((14, 1))
        Ix = np.array([
            i_q,
            i_d,
            i_kq,
            i_f,
            i_kd
        ])[:, np.newaxis]

        P = 50000
        i_dc = beta * np.sqrt(i_q ** 2 + i_d ** 2)

        delta = atan(i_d / i_q) - phi

        iq0 = wbaseline[1 - 1]
        id0 = wbaseline[2 - 1]
        vdc0 = wbaseline[6 - 1]
        nlidc = (iq0 / (sqrt(iq0 ** 2 + id0 ** 2))) * (i_q - iq0) + (id0 / (sqrt(iq0 ** 2 + id0 ** 2))) * (i_d - id0)
        i_dc = beta * sqrt(i_q ** 2 + i_d ** 2)

        delta = atan(i_d / i_q) - phi

        v_q = alhpa * cos(delta) * v_dc
        v_d = alhpa * sin(delta) * v_dc

        V_hat = np.array([v_q,
                          v_d,
                          0.,
                          v_f,
                          0])[:, np.newaxis]

        IA = np.linalg.inv(L) @ (-(R_hat + Omega)) @ Ix + np.linalg.inv(L) @ V_hat

        dxdt[1 - 1] = IA[-1 + 1, :]
        dxdt[2 - 1] = IA[-1 + 2, :]
        dxdt[3 - 1] = IA[-1 + 3, :]
        dxdt[4 - 1] = IA[-1 + 4, :]
        dxdt[5 - 1] = IA[-1 + 5, :]

        dxdt[5] = 1 / C_1 * (i_dc - i_L1)

        dxdt[7 - 1] = 1 / L_1 * (v_dc - i_L1 * R_1 - v_cgen)
        dxdt[8 - 1] = 1 / C_2 * (i_L1 - (1 / R_2) * v_cgen - i_load)
        dxdt[9 - 1] = 1 / L_l * (v_cgen - i_load * R_l - v_Cf1)

        dxdt[10 - 1] = 1 / C_f1 * (i_load - i_CP1)
        dxdt[11 - 1] = 1 / L_CP1 * (v_Cf1 - R_CP1 * i_CP1 - v_Cf2)
        dxdt[12 - 1] = 1 / C_f2 * (i_CP1 - P / v_Cf2)

        tracking_err = K_p * (v_ref - v_cgen)
        u_PI = tracking_err + K_i * int_tracking_err

        dxdt[13 - 1] = (-1 / tau) * v_f + (1 / tau) * u_PI
        dxdt[14 - 1] = tracking_err

        return dxdt
