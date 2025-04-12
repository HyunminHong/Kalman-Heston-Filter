import numpy as np

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
    beta = np.array([-0.5, 1.0]).reshape(-1, 1)   # shape (2,1)
    # Set sigma as a column vector (2,1).
    sigma = np.array([1.0, _sigma]).reshape(-1, 1)  # shape (2,1)
    
    # Precompute elementwise product (sigma ⊙ rho) ensuring column vector shape.
    sigma_rho = sigma * rho  # shape (2,1)

    V_pred = np.zeros(T)
    P_pred = np.zeros(T)
    V_filt = np.zeros(T)
    P_filt = np.zeros(T)
    
    # Initialize state with (possibly burnin-updated) initial conditions.
    V_filt_current = V0 # V_{0|0}
    P_filt_current = P0 # P_{0|0}
    
    for t in range(T):
        if t == 0:
            V_pred_prev = V0
            V_filt_prev = V0
            P_pred_prev = P0  # Use the initialized state rather than P_pred[-1].
        else:
            V_pred_prev = V_pred[t-1]
            P_pred_prev = P_pred[t-1]         # Predicted covariance from the previous step.
        
        ## Prediction Step
        H = (beta @ beta.T) * P_pred_prev + (sigma @ sigma.T) * V_filt_prev
        if np.linalg.cond(H) > 1e10:
            H_inv = np.linalg.pinv(H)
        else:
            H_inv = np.linalg.inv(H)

        # Compute Kalman gain using transpose so that K is (1,2).
        K = np.sqrt(max(V_filt_prev, 0)) * (sigma_rho.T @ H_inv)
        
        # The prediction residual using the previous filtered state.
        residual_pred = Y[t] - mu - beta * V_pred_prev
        V_pred[t] = kappa * theta + (1.0 - kappa) * V_filt_current + xi * np.sqrt(max(V_filt_current, 0)) * (K @ residual_pred).item()
        bracket = 1.0 - np.sqrt(max(V_filt_prev, 0)) * (K @ sigma_rho).item()
        P_pred[t] = (1.0 - kappa)**2 * P_filt_current + (xi**2) * V_filt_current * bracket
        
        ## Update Step
        residual_update = Y[t+1] - mu - beta * V_pred[t]
        S = (beta @ beta.T) * P_pred[t] + (sigma @ sigma.T) * V_filt_current
        if np.linalg.cond(S) > 1e10:
            S_inv = np.linalg.pinv(S)
        else:
            S_inv = np.linalg.inv(S)
        
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

if __name__ == '__main__':
    np.random.seed(42)
    T = 100
    # Generate synthetic data for T time steps.
    returns = 0.001 + 0.01 * np.random.randn(T)  # e.g., daily returns
    RVs = 0.04 + 0.005 * np.random.randn(T)       # e.g., realized variances
    
    # Parameters for the measurement equations and the latent variance.
    mu = np.array([0.001, 0.0])          # Intercept: for returns and RVs respectively.
    sigma = np.array([1.0, 0.5])         # Noise scaling for returns and RV.
    rho = np.array([-0.7, 0.0])            # rho[0] (leverage effect) and rho[1]
    kappa = 0.5                          # Mean reversion speed.
    theta = 0.04                         # Long-run variance.
    xi = 0.5                             # Volatility-of-volatility.
    V0 = 0.04                            # Initial variance.
    P0 = 0.001                           # Initial covariance of latent variance.
    
    # Run the Kalman-like filter.
    V_pred, P_pred, V_filt, P_filt = kalman_like_heston_filter_2d(
        returns, RVs, mu, sigma, rho, kappa, theta, xi, V0, P0
    )
    
    print("Predicted latent variances (V_pred):\n", V_pred)
    print("\nFiltered latent variances (V_filt):\n", V_filt)
