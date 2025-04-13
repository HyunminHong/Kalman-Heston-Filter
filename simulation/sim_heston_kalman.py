import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Point to the parent folder (KALMAN-HESTON-FILTER)
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Now Python sees "model" as a valid package
from model.kalman_heston_filter import kalman_like_heston_filter, estimate_params_qmle
from model.heston_mod import heston_DGP

def plot_heston_vs_kalman(V_true, V_pred, V_filt, title=None):
    """
    Plot the simulated Heston Model latent process alongside the Kalman filter approximations.
    
    Parameters:
    V_true : ndarray
        The true latent process from the simulation. (Typically length T+1)
    V_pred : ndarray
        One-step-ahead predictions from the Kalman-like filter (length T).
    V_filt : ndarray
        Filtered estimates from the Kalman-like filter (length T).
    title : str, optional
        Overall title for the plots.
    
    Notes:
    Since the simulation returns V_true with T+1 values (from time 0 to T)
    and the filter is computed over T observations, we align by comparing the 
    first T values of V_true with the filter outputs.
    """
    # Determine the number of filter outputs
    n = len(V_filt)
    # Create a time vector for the filter (0 to T-1)
    time = np.arange(n)
    
    # Create two subplots side by side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: True latent state vs. Kalman prediction and filtered estimates
    axs[0].plot(time, V_true[:n], label="True V", color="blue")
    # axs[0].plot(time, V_pred, label="Kalman Prediction", linestyle="--", color="red")
    axs[0].plot(time, V_filt, label="Kalman Filter", linestyle="-.", color="green")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Latent Variance V")
    axs[0].set_title("Simulated Heston Model vs. Kalman Approximation")
    axs[0].legend()
    
    # Right plot: Filtering error over time
    # (difference between the true latent state and the filtered estimate)
    error = V_true[:n] - V_filt
    axs[1].plot(time, error, label="Error (True V - Filter)", color="purple")
    axs[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Error")
    axs[1].set_title("Filtering Error Over Time")
    axs[1].legend()
    
    if title is not None:
        plt.suptitle(title)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # True parameters for the simulated Heston model (unknown in estimation):
    true_params = {
        "alpha": 0.1,
        "beta": 0.9, 
        "gamma": 0.3,
        "delta": 0.2
    }
    T = 1000           # Number of observations for estimation.
    burn_in = 0      # Burn-in period.

    # Arbitrary diffusion starting value (since we don't know the true V):
    V0_initial = 10.0   # Use this as initial guess for the latent state in filtering
    P0_init = 0.1       # Initial filtering uncertainty

    # Simulate the Heston model (the "real" data)
    # noise_t = {'df': 2.5}
    # noise_par = {'alpha': 2.5}
    V_series, y_series = heston_DGP(T, **true_params, V0=V0_initial, seed=42, burn_in=burn_in)

    # QMLE Estimation: Start with an initial guess reflecting a diffusion process.
    init_guess = np.array([0.5, 0.5, 0.5, 0.5])
    bounds = [
        (1e-8, None),      # alpha >= 0
        (1e-8, 1 - 1e-8),  # 0 < beta < 1
        (1e-8, None),      # gamma >= 0
        (1e-8, None)       # delta >= 0
    ]

    result = estimate_params_qmle(y_series, V0=V0_initial, P0=P0_init, init_params=init_guess)
    estimated_params = result.x

    # Now use the estimated parameters in the Kalman filter
    V_pred, P_pred, V_filt, P_filt = kalman_like_heston_filter(y_series, *estimated_params, V0=V0_initial, P0=P0_init)

    # Print QMLE results if needed
    print("Estimated parameters:", estimated_params)

    # Plot the simulated Heston process vs. the Kalman approximation (using estimated parameters)
    plot_heston_vs_kalman(V_series, V_pred, V_filt, title="Simulated Heston vs. Kalman Approximation")
