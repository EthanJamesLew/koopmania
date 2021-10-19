import numpy as np
from koopmania.dmd import dmd, get_linear_transform, learned_dmd_sys, get_traj
from koopmania.observable import IdentityObservable, KoopmanObservable
import functools


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
                 rank = 10,
                 sampling_period = 1.0):
        super().__init__()
        self.g = observable_fcn.obs_fcn
        self.gd = observable_fcn.obs_grad
        self.rank = rank
        self.sampling_period = sampling_period
        self._X, self._Xn = None, None

    def fit(self, X: np.ndarray, Xn: np.ndarray) -> None:
        self.dynamics = learned_dmd_sys(X, Xn, self.g, self.gd, self.sampling_period, self.rank)
        self._X, self._Xn = X, Xn
        self.has_trained = True

    @requires_trained
    def predict_derivative(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.dynamics(x) for x in X])

    @requires_trained
    def predict(self, X: np.ndarray) -> np.ndarray:
        """FIXME: TODO: higher order solving?"""
        return get_traj(self.dynamics, X.flatten(), tmax=self.sampling_period, sampling_period=self.sampling_period)

    @property
    @requires_trained
    def linear_continuous_transform(self) -> np.ndarray:
        return get_linear_transform(self._X, self._Xn, self.g, self.sampling_period, self.rank, True)

    @property
    @requires_trained
    def linear_transform(self) -> np.ndarray:
        return get_linear_transform(self._X, self._Xn, self.g, self.sampling_period, self.rank, False)
