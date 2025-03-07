import numpy as np
from scipy.optimize import minimize

def kalman_like_heston_filter(
    y: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    V0: float,
    P0: float
):
    """
    Kalman-like Heston filter (approximation) for the discrete-time model.
    
    Follows the approximation in Equation (3–10) from ID39 - Heston.
    
    Parameters:
    y : ndarray of shape (T,)
        Observed time series.
    alpha, beta, gamma, delta : float
        Model parameters.
    V0 : float
        Initial filtered mean for V.
    P0 : float
        Initial filtered variance for V.
    
    Returns:
    V_pred : ndarray of shape (T,)
        One-step-ahead predicted means (V_{t|t-1}).
    P_pred : ndarray of shape (T,)
        One-step-ahead predicted variances.
    V_filt : ndarray of shape (T,)
        Filtered means (V_{t|t}).
    P_filt : ndarray of shape (T,)
        Filtered variances.
    """
    T = len(y)
    V_pred = np.zeros(T)
    P_pred = np.zeros(T)
    V_filt = np.zeros(T)
    P_filt = np.zeros(T)
    
    V_filt_prev = V0
    P_filt_prev = P0
    
    for t in range(T):
        # Prediction step
        V_pred[t] = alpha + beta * V_filt_prev
        P_pred[t] = beta**2 * P_filt_prev + gamma**2 * V_filt_prev
        
        # Approximate measurement variance = delta^2 * V_pred[t]
        R_t = delta**2 * max(V_pred[t], 0.0)
        
        # Kalman gain
        denom = P_pred[t] + R_t
        K_t = 0.0 if denom < 1e-12 else P_pred[t] / denom
        
        # Update step
        V_filt[t] = V_pred[t] + K_t * (y[t] - V_pred[t])
        P_filt[t] = (1 - K_t) * P_pred[t]
        
        # Prepare for next iteration
        V_filt_prev = V_filt[t]
        P_filt_prev = P_filt[t]
    
    return V_pred, P_pred, V_filt, P_filt

def kalman_like_heston_loglike(params: np.ndarray, y: np.ndarray, V0: float, P0: float) -> float:
    """
    Compute the log-likelihood for the approximate Kalman-like Heston filter.
    
    Under the Gaussian assumption:
        y_t ~ N(V_pred, P_pred + delta^2 * V_pred).
    
    Parameters:
    params : ndarray
        Array of model parameters [alpha, beta, gamma, delta].
    y : ndarray of shape (T,)
        Observed data.
    V0 : float
        Initial mean of V.
    P0 : float
        Initial variance of V.
    
    Returns:
    ll : float
        The total log-likelihood.
    """
    alpha, beta, gamma, delta = params
    V_pred, P_pred, _, _ = kalman_like_heston_filter(y, alpha, beta, gamma, delta, V0, P0)
    
    T = len(y)
    ll = 0.0
    for t in range(T):
        Sigma_t = P_pred[t] + delta**2 * max(V_pred[t], 0.0)
        if Sigma_t <= 1e-12:
            return -1e15
        resid = y[t] - V_pred[t]
        ll_t = -0.5 * np.log(2.0 * np.pi * Sigma_t) - 0.5 * (resid**2 / Sigma_t)
        ll += ll_t
    return ll

def estimate_params_qmle(y: np.ndarray, V0: float, P0: float, init_params: np.ndarray = None, bounds=None):
    """
    Estimate the model parameters via Quasi-Maximum Likelihood Estimation.
    
    Parameters:
    y : ndarray of shape (T,)
        Observed data.
    V0 : float
        Initial mean of V.
    P0 : float
        Initial variance of V.
    init_params : ndarray of shape (4,), optional
        Initial guess for [alpha, beta, gamma, delta].
    bounds : list of (float, float), optional
        Bounds for the optimizer.
    
    Returns:
    result : OptimizeResult
        The optimization result from scipy.optimize.minimize.
    """
    if init_params is None:
        init_params = np.array([0.1, 0.8, 0.2, 0.2])
    if bounds is None:
        bounds = [
            (1e-8, None),      # alpha >= 0
            (1e-8, 1 - 1e-8),  # 0 < beta < 1
            (1e-8, None),      # gamma >= 0
            (1e-8, None)       # delta >= 0
            ]

    def neg_loglike(p):
        return -kalman_like_heston_loglike(p, y, V0, P0)
    
    result = minimize(neg_loglike, init_params, method='L-BFGS-B', bounds=bounds)
    return result