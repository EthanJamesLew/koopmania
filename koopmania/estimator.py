import numpy as np
from koopmania.dmd import dmd, get_linear_transform, learned_dmd_sys
from koopmania.trajectory import get_traj
from koopmania.observable import IdentityObservable, KoopmanObservable
from koopmania.system import ContinuousSystem, KoopmanContinuousSystem
import functools
from scipy.linalg import logm


def requires_trained(f):
    @functools.wraps(f)
    def inner(estimator, *args, **kwargs):
        if estimator.has_trained:
            return f(estimator, *args, **kwargs)
        else:
            raise RuntimeError(f"method {f} requires that the estimator {estimator} has been trained")
    return inner


class Estimator:
    """TODO: FIXME: maybe replace with scikit estimator"""
    def __init__(self):
        self.has_trained = False

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class KoopmanSystemEstimator(Estimator):
    def __init__(self,
                 observable_fcn: KoopmanObservable,
                 sampling_period: float,
                 rank = 10):
        super().__init__()
        self.obs = observable_fcn
        self.g = observable_fcn.obs_fcn
        self.gd = observable_fcn.obs_grad
        self.rank = rank
        self.sampling_period = sampling_period
        self._X, self._Xn = None, None
        self._system, self._A, self._B = None, None, None

    def fit(self, X: np.ndarray, Xn: np.ndarray) -> None:
        Yc, Ycp = np.hstack([self.g(x) for x in X.T]), np.hstack([self.g(x) for x in Xn.T])
        mucy, Phicy = dmd(Yc, Ycp, r=self.rank)
        self._A = Phicy @ np.diag(mucy) @ np.linalg.pinv(Phicy)
        self._B = logm(self._A) / self.sampling_period
        self._system = KoopmanContinuousSystem(X.shape[0], np.real(self._B), self.obs)
        self.has_trained = True

    @requires_trained
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._system.trajectory(x, (0.0, self.sampling_period), self.sampling_period)[-1] for x in X])

    @property
    @requires_trained
    def system(self) -> KoopmanContinuousSystem:
        return self._system
