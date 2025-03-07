import numpy as np
from draw_noise import draw_noise

def simulate_heston(
    T: int, 
    alpha: float, 
    beta: float, 
    gamma: float, 
    delta: float, 
    V0: float = 1.0, 
    seed: int = None, 
    noise_dist: str = 'normal', 
    noise_params: dict = None, 
    burn_in: int = 0
):
    """
    Simulate from the discrete-time Heston model with an optional burn-in period.
    
    Model:
        V_{t+1} = alpha + beta * V_t + gamma * sqrt(V_t) * eta_{t+1},
        y_t     = V_t + delta * sqrt(V_t) * eps_t,
    where eta_{t+1} and eps_t are i.i.d. draws from the specified noise distribution
    (with mean 0, variance 1).
    
    Parameters:
    T : int
        Number of observation points to return (after burn-in).
    alpha, beta, gamma, delta : float
        Model parameters.
    V0 : float
        Initial latent variance.
    seed : int, optional
        Random seed for reproducibility.
    noise_dist : str, optional
        Noise distribution to use: 'normal', 't', or 'pareto'. Default 'normal'.
    noise_params : dict, optional
        Parameters for the chosen noise distribution.
    burn_in : int, optional
        Number of initial observations to discard (burn-in period).
    
    Returns:
    V : ndarray of shape (T+1,)
        The latent variance series corresponding to the returned observations.
    y : ndarray of shape (T,)
        The observations.
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_steps = T + burn_in
    V_full = np.zeros(total_steps + 1)
    y_full = np.zeros(total_steps)
    
    V_full[0] = V0
    for t in range(total_steps):
        # Draw shocks (eta, eps) from the specified distribution
        eta = draw_noise(noise_dist, noise_params)
        eps = draw_noise(noise_dist, noise_params)
        
        # Next latent variance
        V_full[t+1] = alpha + beta * V_full[t] + gamma * np.sqrt(max(V_full[t], 0.0)) * eta
        
        # Observation
        y_full[t] = V_full[t] + delta * np.sqrt(max(V_full[t], 0.0)) * eps
    
    # Discard the burn-in period
    V = V_full[burn_in:]
    y = y_full[burn_in:]
    
    return V, y