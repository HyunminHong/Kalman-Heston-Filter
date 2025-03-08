import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import pareto

def draw_noise(dist_type: str, dist_params=None):
    """
    Draw a single sample from a chosen distribution with mean=0, var=1.
    
    Parameters:
    dist_type : str
        'normal', 't', or 'pareto'
    dist_params : dict, optional
        Dictionary of distribution parameters:
          - for 't',  use dist_params.get('df', 5) for degrees of freedom
          - for 'pareto', use dist_params.get('alpha', 3) for the shape parameter
        If None, defaults are used.
    
    Returns:
    float
        A single random draw with mean 0 and variance 1.
    """
    if dist_params is None:
        dist_params = {}
    
    if dist_type.lower() == 'normal':
        # Standard normal
        return np.random.randn()
    
    elif dist_type.lower() == 't':
        df = dist_params.get('df', 5)
        if df <= 2:
            raise ValueError("For a t-distribution with df <= 2, the variance is infinite. "
                             "Please use df > 2 if you want to standardize to var=1.")
        # Student-t with df>2 has variance df/(df-2). We divide by sqrt(...) to get var=1.
        x = t_dist.rvs(df, size=1)
        return x[0] / np.sqrt(df/(df-2))
    
    elif dist_type.lower() == 'pareto':
        alpha_param = dist_params.get('alpha', 3)
        if alpha_param <= 2:
            raise ValueError("Pareto with alpha <= 2 has infinite variance; "
                             "cannot standardize to var=1. Use alpha > 2.")
        # Standard (scale=1) Pareto has mean = alpha/(alpha-1) (for alpha>1)
        # and variance = alpha / ((alpha-1)^2 (alpha-2)) (for alpha>2).
        x = pareto.rvs(alpha_param, size=1)
        mean_theo = alpha_param/(alpha_param - 1)
        var_theo  = alpha_param / ((alpha_param - 1)**2 * (alpha_param - 2))
        x_centered = x[0] - mean_theo
        x_standardized = x_centered / np.sqrt(var_theo)
        return x_standardized
    
    else:
        raise ValueError(f"Unknown dist_type '{dist_type}'. Choose 'normal', 't', or 'pareto'.")

def heston_DGP(
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