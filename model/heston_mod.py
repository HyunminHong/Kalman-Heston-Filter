import numpy as np

def heston_DGP(T: int, alpha: float, beta: float, gamma: float, delta: float, V0: float = 1.0, seed: int = None, burn_in: int = 0):
    if seed is not None:
        np.random.seed(seed)
    
    total_steps = T + burn_in
    V_full = np.zeros(total_steps + 1)
    y_full = np.zeros(total_steps)
    
    V_full[0] = V0
    for t in range(total_steps):
        # Draw shocks (eta, eps) from the specified distribution
        eta = np.random.randn()
        eps = np.random.randn()
        
        # Next latent variance
        V_full[t+1] = alpha + beta * V_full[t] + gamma * np.sqrt(max(V_full[t], 0.0)) * eta
        
        # Observation
        y_full[t] = V_full[t] + delta * np.sqrt(max(V_full[t], 0.0)) * eps
    
    # Discard the burn-in period
    V = V_full[burn_in:]
    y = y_full[burn_in:]
    
    return V, y