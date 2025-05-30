import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import scipy.special as spec

from scipy.optimize import fmin, fmin_bfgs
from scipy.stats import norm
from numpy.random import gamma
from filterpy.monte_carlo import systematic_resample, stratified_resample # type: ignore

class PFHeston(object):
    def __init__(self, y, N=1000, dt=1/252, is_log=False):
        self.y = y
        self.logS0 = np.log(y[0]) if not is_log else y[0]
        self.N = N # num particles
        self.dt = dt

    def filter(self, params, is_bounds=True, simple_resample=False, predict_obs=False):
        """
        Performs sequential monte-carlo sampling particle filtering
        Note: Currently only supports a bound of parameters
        """
        y = self.y
        N = self.N

        if not is_bounds: # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
            params_states = np.tile(np.array([mu, kappa, theta, sigma, rho])[:, None], (1, N))
        else:
            # initialize param states, N particles for each param sampled uniformly
            v0 = params[-1] # params is shape [(lb, ub)_1,...,k, v0]
            params_states = self._init_parameter_states(N, params[:-1])

        observations = np.zeros(len(y))
        hidden = np.zeros(len(y))
        observations[0] = y[0]
        hidden[0] = v0

        # particles = np.maximum(1e-3, self.proposal_sample(self.N, v, dy, params))
        weights = np.array([1/self.N] * self.N)

        # initialize v particles
        particles = norm.rvs(v0, 0.02, N)
        particles = np.maximum(1e-4, particles)

        # storing the estimated parameters each step
        params_steps = np.zeros((len(params)-1, len(y)))
        params_steps.transpose()[0] = np.mean(params_states, axis=1)

        for i in range(1, len(y)):
            dy = y[i] - y[i-1]

            # prediction
            # proposal sample
            x_pred = self.proposal_sample(N, particles, dy, params_states)
            x_pred = np.maximum(1e-3, x_pred)

            # weights
            Li = self.likelihood(y[i], x_pred, particles, y[i-1], params_states)
            I = self.proposal(x_pred, particles, dy, params_states)
            T = self.transition(x_pred, particles, params_states)
            # weights = weights * (Li*T/I)
            # weights = weights/sum(weights)

            eps = 1e-300
            # 1) avoid zero in I
            I_safe = np.clip(I, eps, None)              # ensures every entry ≥ eps
            weights = weights * (Li * T / I_safe)

            # 2) normalize, but guard against zero total
            w_sum = weights.sum()
            if w_sum <= 0 or not np.isfinite(w_sum):
                # degenerate case: reset to uniform
                weights = np.ones_like(weights) / len(weights)
            else:
                weights = weights / w_sum

            # Resampling
            if self._neff(weights) < 0.7*self.N:
                print('resampling since: {}'.format(self._neff(weights)))
                if simple_resample:
                    x_pred, weights, params_states = self._simple_resample(x_pred, weights, params_states)
                else:
                    x_pred, weights, params_states = self._systematic_resample(x_pred, weights, params_states)

            # observation prediction
            if predict_obs:
                y_hat = self.observation_predict(x_pred, particles, y[i-1], np.mean(params_states[0])) # mu is the 0 index
                observations[i] = y_hat
                print("Done with iter: {}".format(i))

            hidden[i] = np.sum(x_pred * weights)
            particles = x_pred
            params_steps.transpose()[i] = np.sum(np.multiply(params_states, weights[np.newaxis, :]), axis=1)

        return (hidden, params_steps, observations) if predict_obs else (hidden, params_steps)

    def obj_likelihood(self, x, dy_next, params):
        if isinstance(params, list): # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = dy_next + (mu-1/2*x)*self.dt
        s = np.sqrt(x*self.dt)
        return norm.pdf(x, m, s)

    def proposal(self, x, x_prev, dy, params):
        if isinstance(params, list): # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = x_prev + kappa*(theta-x_prev)*self.dt + sigma*rho*(dy - (mu-1/2*x_prev)*self.dt)
        s = sigma*np.sqrt(x_prev*(1-rho**2)*self.dt)
        return norm.pdf(x, m, s)

    def likelihood(self, y, x, x_prev, y_prev, params):
        if isinstance(params, list): # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = y_prev + (mu-1/2*x)*self.dt
        s = np.sqrt(x_prev*self.dt)
        return norm.pdf(y, m ,s)

    def transition(self, x, x_prev, params):
        if isinstance(params, list): # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = 1/(1+1/2*sigma*rho*self.dt) * (x_prev + kappa*(theta-x_prev)*self.dt + 1/2*sigma*rho*x_prev*self.dt)
        s = 1/(1+1/2*sigma*rho*self.dt) * sigma * np.sqrt(x_prev*self.dt)
        return norm.pdf(x, m, s)

    def prediction_density(self, y, y_prev, x, mu):
        m = y_prev + (mu-1/2*x)*self.dt
        s = np.sqrt(x*self.dt)
        return norm.pdf(y, m, s)

    def observation_predict(self, x_pred, particles, y_prev, mu):
        y_hat = y_prev + (mu-1/2*x_pred)*self.dt + np.sqrt(particles*self.dt)*norm.rvs()
        py_hat = np.array([np.mean(self.prediction_density(y_hat[k], y_prev, x_pred, mu)) for k in range(len(y_hat))])
        py_hat = py_hat/sum(py_hat)
        return np.sum(py_hat * y_hat)

    def _init_parameter_states(self, N, bounds):
        # initialize param states
        params_states = np.zeros((len(bounds), N))
        b0, b1, b2, b3, b4 = bounds
        params_states[0] = np.random.rand(N)*(b0[1]-b0[0])+b0[0]
        params_states[1] = np.random.rand(N)*(b1[1]-b1[0])+b1[0]
        params_states[2] = np.random.rand(N)*(b2[1]-b2[0])+b2[0]
        params_states[3] = np.random.rand(N)*(b3[1]-b3[0])+b3[0]
        params_states[4] = np.random.rand(N)*(b4[1]-b4[0])+b4[0]
        return params_states

    def proposal_sample(self, N, x_prev, dy, params):
        """
        x_prev is array of particles
        """
        if len(params.shape) < 2: # params is an array of param values, not particles
            mu, kappa, theta, sigma, rho, v0 = self._unwrap_params(params)
        else:
            mu, kappa, theta, sigma, rho = self._unwrap_param_states(params)
        m = x_prev + kappa*(theta-x_prev)*self.dt + sigma*rho*(dy - (mu-1/2*x_prev)*self.dt)
        s = sigma*np.sqrt(x_prev*(1-rho**2)*self.dt)
        return norm.rvs(m, s, N)

    def __simple_resample(self, particles, weights):
        N = len(particles)
        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1. # avoid round-off error
        indexes = np.searchsorted(cumulative_sum, np.random.rand(N))

	    # resample according to indexes
        particles[:] = particles[indexes]
        new_weights = np.ones(len(weights)) / len(weights)
        return particles, new_weights

    def _resample_from_index(self, particles, weights, indexes):
        particles[:] = particles[indexes]
        weights[:] = weights[indexes]
        new_weights = np.ones(len(weights)) / len(weights)
        return particles, new_weights

    def _systematic_resample(self, x_pred, weights, params_states):
        idxs = systematic_resample(weights)
        params_states[0], _ = self._resample_from_index(params_states[0], weights, idxs)
        params_states[1], _ = self._resample_from_index(params_states[1], weights, idxs)
        params_states[2], _ = self._resample_from_index(params_states[2], weights, idxs)
        params_states[3], _ = self._resample_from_index(params_states[3], weights, idxs)
        params_states[4], _ = self._resample_from_index(params_states[4], weights, idxs)
        x_pred, weights = self._resample_from_index(x_pred, weights, idxs)
        return x_pred, weights, params_states

    def _simple_resample(self, x_pred, weights, params_states):
        params_states[0], _ = self.__simple_resample(params_states[0], weights)
        params_states[1], _ = self.__simple_resample(params_states[1], weights)
        params_states[2], _ = self.__simple_resample(params_states[2], weights)
        params_states[3], _ = self.__simple_resample(params_states[3], weights)
        params_states[4], _ = self.__simple_resample(params_states[4], weights)
        x_pred, weights = self.__simple_resample(x_pred, weights)
        return x_pred, weights, params_states

    def _neff(self, weights):
        return 1. / np.sum(np.square(weights))

    def _unwrap_param_states(self, params_states):
        mu = params_states[0]
        kappa = params_states[1]
        theta = params_states[2]
        sigma = params_states[3]
        rho = params_states[4]
        return mu, kappa, theta, sigma, rho

    def _unwrap_params(self, params):
        def periodic_map(x, c, d):
            """
            Periodic Param mapping provided by Prof. Hirsa
            """
            if ((x>=c) & (x<=d)):
                y = x
            else:
                range = d-c
                n = np.floor((x-c)/range)
                if (n%2 == 0):
                    y = x - n*range;
                else:
                    y = d + n*range - (x-c)
            return y
        mu = periodic_map(params[0], 0.01, 1)
        kappa = periodic_map(params[1], 1, 3)
        theta = periodic_map(params[2], 0.001, 0.2)
        sigma = periodic_map(params[3], 1e-3, 0.7)
        rho = periodic_map(params[4], -1, 1)
        v0 = periodic_map(params[5], 1e-3, 0.2) # ensure positive vt
        return mu, kappa, theta, sigma, rho, v0
