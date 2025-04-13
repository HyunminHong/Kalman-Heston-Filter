import numpy as np
from scipy.optimize import minimize

def kalman_like_heston_filter(params: list, y: np.ndarray, V0: float, P0: float):
    kappa, theta, xi, sigma = params

    T = len(y)
    V_pred = np.zeros(T)
    P_pred = np.zeros(T)
    V_filt = np.zeros(T)
    P_filt = np.zeros(T)
    
    V_filt_prev = V0
    P_filt_prev = P0
    
    for t in range(T):
        # Prediction step
        V_pred[t] = kappa * theta + (1 - kappa) * V_filt_prev
        P_pred[t] = (1 - kappa)**2 * P_filt_prev + xi**2 * V_filt_prev
        
        # Approximate measurement variance = delta^2 * V_pred[t]
        R_t = sigma**2 * max(V_pred[t], 0.0)
        
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
    V_pred, P_pred, _, _ = kalman_like_heston_filter(params, y, V0, P0)
    _, _, _, sigma = params
    
    T = len(y)
    ll = 0.0
    for t in range(T):
        Sigma_t = P_pred[t] + sigma**2 * max(V_pred[t], 0.0)
        if Sigma_t <= 1e-12:
            return -1e15
        resid = y[t] - V_pred[t]
        ll_t = -0.5 * np.log(2.0 * np.pi * Sigma_t) - 0.5 * (resid**2 / Sigma_t)
        ll += ll_t
    return ll

def estimate_params_qmle(y: np.ndarray, V0: float, P0: float, init_params: np.ndarray = None, bounds=None):
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

def kalman_like_heston_filter_2d(params, R, RV, V0, P0):
    # Unpack parameters.
    mu, kappa, theta, xi, _sigma, rho1, rho2 = params
    # For the extended filter we have two correlations.
    rho = np.array([rho1, rho2]).reshape(-1, 1)  # shape (2,1)

    # Prepare the measurement vectors.
    # Create a (T,2) array then reshape each row as a column vector (2,1).
    Y_full = np.hstack((R.reshape(-1, 1), RV.reshape(-1, 1)))   # shape (T,2)
    Y = Y_full.reshape(-1, 2, 1) 
    T = Y.shape[0]

    # Set fixed measurement coefficient vector beta as a column vector (2,1).
    mu = np.array([mu, 0]).reshape(-1,1)
    beta = np.array([-0.5, 1.0]).reshape(-1, 1)   # shape (2,1)
    # Set sigma as a column vector (2,1).
    sigma = np.array([1.0, _sigma]).reshape(-1, 1)  # shape (2,1)
    
    # Precompute elementwise product (sigma ⊙ rho) ensuring column vector shape.
    sigma_rho = sigma * rho  # shape (2,1)

    num_steps = T - 1
    V_pred = np.zeros(T)
    P_pred = np.zeros(T)
    V_filt = np.zeros(T)
    P_filt = np.zeros(T)
    
    # Initialize state with (possibly burnin-updated) initial conditions.
    V_filt_current = V0 # V_{0|0}
    P_filt_current = P0 # P_{0|0}
    
    for t in range(num_steps):
        if t == 0:
            V_pred_prev = V0
            P_pred_prev = P0  

            V_filt_prev = V0
        else:
            V_pred_prev = V_pred[t-1]
            P_pred_prev = P_pred[t-1]
        
        ## Prediction Step
        H = (beta @ beta.T) * P_pred_prev + (sigma @ sigma.T) * V_filt_prev
        H_inv = np.linalg.pinv(H) if np.linalg.cond(H) > 1e10 else np.linalg.inv(H)

        K = np.sqrt(max(V_filt_prev, 0)) * (sigma_rho.T @ H_inv) # shape (2,1)
        
        # The prediction residual using the previous filtered state.
        residual_pred = Y[t] - mu - beta * V_pred_prev
        V_pred[t] = kappa * theta + (1.0 - kappa) * V_filt_current + xi * np.sqrt(max(V_filt_current, 0)) * (K @ residual_pred).item()
        bracket = 1.0 - np.sqrt(max(V_filt_prev, 0)) * (K @ sigma_rho).item()
        P_pred[t] = (1.0 - kappa)**2 * P_filt_current + (xi**2) * V_filt_current * bracket
        
        ## Update Step
        residual_update = Y[t+1] - mu - beta * V_pred[t]
        S = (beta @ beta.T) * P_pred[t] + (sigma @ sigma.T) * V_filt_current
        S_inv = np.linalg.pinv(S) if np.linalg.cond(S) > 1e10 else np.linalg.inv(S)

        correction_update = P_pred[t] * (beta.T @ (S_inv @ residual_update)).item()
        V_filt[t] = V_pred[t] + correction_update
        scalar_term = (beta.T @ (S_inv @ beta)).item()
        P_filt[t] = P_pred[t] - (P_pred[t]**2) * scalar_term 
        P_filt[t] = max(P_filt[t], 1e-12)
        
        # Update state for next iteration.
        V_filt_prev = V_filt_current

        V_filt_current = V_filt[t]
        P_filt_current = P_filt[t]
    
    return V_pred, P_pred, V_filt, P_filt

# def kalman_like_heston_filter_2d(params, R, RV, V0, P0, dt):
#     # Unpack parameters.
#     mu, kappa, theta, xi, _sigma, rho1, rho2 = params

#     Y_full = np.hstack((R.reshape(-1, 1), RV.reshape(-1, 1))) # shape (T,2)
#     Y = Y_full.reshape(-1, 2, 1) 
#     T = Y.shape[0]

#     mu = np.array([mu*dt, 0]).reshape(-1, 1) # shape (2,1)
#     beta = np.array([-0.5*dt, dt]).reshape(-1, 1) # shape (2,1)
#     sigma = np.array([np.sqrt(dt), _sigma*np.sqrt(dt)]).reshape(-1, 1) # shape (2,1)
#     rho = np.array([rho1, rho2]).reshape(-1, 1)  # shape (2,1)

#     # Precompute elementwise product (sigma ⊙ rho) ensuring column vector shape.
#     sigma_rho = sigma * rho  # shape (2,1)

#     num_steps = T - 1
#     V_pred = np.zeros(T)
#     P_pred = np.zeros(T)
#     V_filt = np.zeros(T)
#     P_filt = np.zeros(T)
    
#     # Initialize state with (possibly burnin-updated) initial conditions.
#     V_filt[0] = V0 # V_{0|0}
#     P_filt[0] = P0 # P_{0|0}
    
#     for t in range(num_steps):
#         if t == 0:
#             V_pred_prev = V0 # V_{0|-1}
#             P_pred_prev = P0 # P_{0|-1}

#             V_filt_prev = V0 # V_{-1|-1}

#             K = np.zeros((1,2))
#         else:
#             V_pred_prev = V_pred[t-1] # V_{t|t-1}
#             P_pred_prev = P_pred[t-1] # P_{t|t-1}
        
#             ## Prediction Step
#             H = (beta @ beta.T) * P_pred_prev + (sigma @ sigma.T) * V_filt_prev
#             H_inv = np.linalg.pinv(H) if np.linalg.cond(H) > 1e10 else np.linalg.inv(H)

#             K = np.sqrt(max(V_filt_prev, 0)) * (sigma_rho.T @ H_inv) # shape (1,2)
        
#         residual_pred = Y[t] - mu - beta * V_pred_prev # shape (2,1)
#         V_pred[t] = kappa * theta + (1.0 - kappa) * V_filt[t] + xi * np.sqrt(max(V_filt[t], 0)) * (K @ residual_pred)
#         bracket = 1.0 - np.sqrt(max(V_filt_prev, 0)) * (K @ sigma_rho)
#         P_pred[t] = (1.0 - kappa)**2 * P_filt[t] + (xi**2) * V_filt[t] * bracket
        
#         ## Update Step
#         residual_update = Y[t+1] - mu - beta * V_pred[t] # shape (2,1)
#         S = (beta @ beta.T) * P_pred[t] + (sigma @ sigma.T) * V_filt[t] # shape (2,2)
#         S_inv = np.linalg.pinv(S) if np.linalg.cond(S) > 1e10 else np.linalg.inv(S) # shape (2,2)

#         correction_update = P_pred[t] * (beta.T @ S_inv @ residual_update)  # shape (1,1)
#         V_filt[t+1] = V_pred[t] + correction_update
#         scalar_term = (beta.T @ S_inv @ beta)
#         P_filt[t+1] = P_pred[t] - (P_pred[t] * scalar_term * P_pred[t])
#         P_filt[t+1] = max(P_filt[t], 1e-12)
        
#         # Update state for next iteration.
#         V_filt_prev = V_filt[t+1]
    
#     return V_pred, P_pred, V_filt, P_filt