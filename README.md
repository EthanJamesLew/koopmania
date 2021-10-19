# Koopmania: Library for Data-Driven Koopman Analyses

This is a (under development) little library to help me with research involving Koopman
operators. It features
* Estimators - scikit-like estimators to fit systems from data. These can be
continuous or discrete time systems.
* Observables - a way of creating Koopman observables easily. These can be
specified symbolically or numerically.

## Example Usage

```python
import koopmania as km


# learn a system from trajectory data X, Xn
obs = km.QuadraticObservable(2)
est = km.KoopmanSystemEstimator(obs, sampling_period=0.1)
est.fit(X, Xn)


# predict the next state (after one sampling period)
est.predict(initial_value) 


# get trajectory from the learned system
trajectory = est.system.trajectory(initial_value, (0.0, 10.0), sampling_period=0.1)
```