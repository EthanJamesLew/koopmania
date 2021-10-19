from scipy.linalg import logm
import scipy.integrate as scint
import numpy as np


def dmd(X, Xp, r=2):
    """simple dynamic mode decomposition (dmd) implementation"""
    # compute Atilde
    U, Sigma, V = np.linalg.svd(X, False)
    U, Sigma, V = U[:, :r], np.diag(Sigma)[:r, :r], V.conj().T[:, :r]
    Atilde = U.conj().T @ Xp @ V @ np.linalg.inv(Sigma)

    # eigenvalues of Atilde are same as A
    D, W = np.linalg.eig(Atilde)

    # recover U.shape[0] eigenfunctions of A
    phi = Xp @ V @ (np.linalg.inv(Sigma)) @ W
    return D, phi


def evolve_modes(mu, Phi, ic, sampling_period, tmax=20.0):
    """time evolve modes of the DMD"""
    b = np.linalg.pinv(Phi) @ np.array(ic)
    t = np.linspace(0, tmax, int(tmax/sampling_period))
    dt = t[2] - t[1]
    psi = np.zeros([len(ic), len(t)], dtype='complex')
    for i,ti in enumerate(t):
        psi[:,i] = np.multiply(np.power(mu, ti / dt), b)
    return psi


def learned_sys(Xc, Xcp, p, g, gradg, sampling_period):
    Yc, Ycp = np.hstack([g(x, p=p) for x in Xc.T]), np.hstack([g(x, p=p) for x in Xcp.T])
    A = Ycp @ np.linalg.pinv(Yc)
    B = logm(A) / sampling_period
    def sys(t, x):
        return np.real(np.linalg.pinv(gradg(x, p=p)) @ B @ g(x, p=p)).flatten()
    return sys


def learned_dmd_sys(Xc, Xcp, g, gradg, sampling_period, r=30):
    Yc, Ycp = np.hstack([g(x) for x in Xc.T]), np.hstack([g(x) for x in Xcp.T])
    mucy, Phicy = dmd(Yc, Ycp, r=r)
    A = Phicy @ np.diag(mucy) @ np.linalg.pinv(Phicy)
    B = logm(A) / sampling_period
    def sys(t, x):
        return np.real(np.linalg.pinv(gradg(x)) @ B @ g(x)).flatten()
    return sys


def get_linear_transform(Xc, Xcp, g, sampling_period, r=30, continuous=True):
    Yc, Ycp = np.hstack([g(x) for x in Xc.T]), np.hstack([g(x) for x in Xcp.T])
    mucy, Phicy = dmd(Yc, Ycp, r)
    A = Phicy @ np.diag(mucy) @ np.linalg.pinv(Phicy)
    if continuous:
        B = logm(A) / sampling_period
        return B
    else:
        return A


