import numpy as np
from typing import Tuple, Callable
import scipy.integrate as scint
import koopmania.observable as kobs


class ContinuousSystem:
    def trajectory(self, initial_state: np.ndarray,
                   tspan: Tuple[float, float],
                   sampling_period: float = 0.1) -> np.ndarray:
        sol = scint.solve_ivp(self.gradient,
                              tspan,
                              initial_state,
                              t_eval=np.arange(0, tspan[-1], sampling_period))
        return sol.y

    def gradient(self, time: float, initial_state: np.ndarray) -> np.ndarray:
        pass

    @property
    def dimension(self) -> int:
        raise NotImplementedError


class GradientContinuousSystem(ContinuousSystem):
    def __init__(self, dimension: int, gradient_fcn: Callable[[float, np.ndarray], np.ndarray]):
        self._grad_fcn = gradient_fcn
        self._dimension = dimension

    def gradient(self, time: float, initial_state: np.ndarray) -> np.ndarray:
        return self._grad_fcn(time, initial_state)

    @property
    def dimension(self) -> int:
        return self._dimension


class LinearContinuousSystem(ContinuousSystem):
    def __init__(self, A: np.ndarray):
        self._transform = A

    def gradient(self, time: float, initial_state: np.ndarray) -> np.ndarray:
        return (self._transform @ np.atleast_2d(initial_state).T).flatten()

    @property
    def linear_transform(self):
        return self._transform

    @property
    def dimension(self) -> int:
        return self._transform.shape[0]


class KoopmanContinuousSystem(LinearContinuousSystem):
    def __init__(self, dimension: int, A: np.ndarray, observables: kobs.KoopmanObservable):
        super(KoopmanContinuousSystem, self).__init__(A)
        self._dimension = dimension
        self._obs = observables

    def gradient(self, time: float, initial_state: np.ndarray) -> np.ndarray:
        rhs = super(KoopmanContinuousSystem, self).gradient(time, self._obs.obs_fcn(initial_state).flatten())
        return np.real(np.linalg.pinv(self._obs.obs_grad(initial_state)) @ np.atleast_2d(rhs).T).flatten()

    @property
    def dimension(self) -> int:
        return self._dimension