import numpy as np
import scipy.integrate as scint


def get_traj(system, iv, tmax=1.0, sampling_period=0.1):
    """given a system with the signature f(t, x), return a time evolution
    trajectory at initial value iv"""
    sol = scint.solve_ivp(system,
                          [0, tmax],
                          iv,
                          t_eval=np.arange(0, tmax, sampling_period))
    return sol.y


def get_dataset(ivs, sys, tmax, sampling_period):
    Xt, Xtp = [], []
    for iv in ivs:
        t = get_traj(sys, iv, tmax=tmax, sampling_period=sampling_period)
        x = t[:, :-1]
        xn = t[:, 1:]
        Xt.append(x)
        Xtp.append(xn)

    Xt = np.hstack(Xt)
    Xtp = np.hstack(Xtp)
    return Xt, Xtp
