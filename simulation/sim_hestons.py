import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from model.kalman_heston import kalman_like_heston_filter
from model.heston import heston_DGP

def plot_heston_comparison(V_true_x, V_pred_x, V_filt_x, V_true_y, V_pred_y, V_filt_y, dist_x: str, dist_y: str):
    """
    Create a 2x2 figure to compare Heston simulation+filter results
    under two different noise distributions, dist_x and dist_y.
    
    Parameters:
    V_true_x : np.ndarray
        True variance series for distribution X.
    V_pred_x : np.ndarray
        Predicted variance (Kalman "prediction") for distribution X.
    V_filt_x : np.ndarray
        Filtered variance (Kalman "update") for distribution X.
    V_true_y : np.ndarray
        True variance series for distribution Y.
    V_pred_y : np.ndarray
        Predicted variance (Kalman "prediction") for distribution Y.
    V_filt_y : np.ndarray
        Filtered variance (Kalman "update") for distribution Y.
    dist_x : str
        Label/name of distribution X (e.g. "Normal", "t", "Pareto").
    dist_y : str
        Label/name of distribution Y (e.g. "Normal", "t", "Pareto").
    title : str, optional
        Main title for the entire figure. If None, a default title is used.
    """
    title = f"Comparison of Heston Simulation: {dist_x} vs. {dist_y} Noise"
    
    # Determine lengths
    T_x = len(V_filt_x)
    T_y = len(V_filt_y)

    # Create the figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    
    # ---- TOP ROW: dist_x ----
    time_x = np.arange(T_x)
    # Left: True vs. filter
    axs[0, 0].plot(time_x, V_true_x[:T_x], label=f"True V ({dist_x})", color="blue")
    axs[0, 0].plot(time_x, V_pred_x, label="Kalman Pred", linestyle="--", color="red")
    axs[0, 0].plot(time_x, V_filt_x, label="Kalman Filter", linestyle="-.", color="green")
    axs[0, 0].set_title(f"{dist_x} Noise: Heston vs. Kalman")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Variance V")
    axs[0, 0].legend()
    
    # Right: Error
    error_x = V_true_x[:T_x] - V_filt_x
    axs[0, 1].plot(time_x, error_x, color="purple", label="Error (True - Filter)")
    axs[0, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[0, 1].set_title(f"{dist_x} Noise: Filtering Error")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Error")
    axs[0, 1].legend()
    
    # ---- BOTTOM ROW: dist_y ----
    time_y = np.arange(T_y)
    # Left: True vs. filter
    axs[1, 0].plot(time_y, V_true_y[:T_y], label=f"True V ({dist_y})", color="blue")
    axs[1, 0].plot(time_y, V_pred_y, label="Kalman Pred", linestyle="--", color="red")
    axs[1, 0].plot(time_y, V_filt_y, label="Kalman Filter", linestyle="-.", color="green")
    axs[1, 0].set_title(f"{dist_y} Noise: Heston vs. Kalman")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Variance V")
    axs[1, 0].legend()
    
    # Right: Error
    error_y = V_true_y[:T_y] - V_filt_y
    axs[1, 1].plot(time_y, error_y, color="purple", label="Error (True - Filter)")
    axs[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[1, 1].set_title(f"{dist_y} Noise: Filtering Error")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Error")
    axs[1, 1].legend()
    
    # Overall title
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # 1) True parameters for the Heston model
    true_params = {
        "alpha": 0.1,
        "beta": 0.9, 
        "gamma": 0.3,
        "delta": 0.2
    }
    T = 1000
    burn_in = 200
    seed = 42

    # 2) Simulate under Normal noise
    V_series_normal, y_series_normal = heston_DGP(T, **true_params, V0=1.0, seed=seed, noise_dist='normal', burn_in=burn_in)
    # Kalman-like filter with the true parameters
    V_pred_norm, P_pred_norm, V_filt_norm, P_filt_norm = kalman_like_heston_filter(y_series_normal, **true_params, V0=1.0, P0=0.1)

    # 3) Simulate under t-distributed noise
    V_series_t, y_series_t = heston_DGP(T, **true_params, V0=1.0, seed=seed, noise_dist='t', burn_in=burn_in)
    # Filter with the same (true) parameters
    V_pred_t, P_pred_t, V_filt_t, P_filt_t = kalman_like_heston_filter(y_series_t, **true_params, V0=1.0, P0=0.1)

    # 4) Simulate under Pareto noise
    V_series_pareto, y_series_pareto = heston_DGP(T, **true_params, V0=1.0, seed=seed, noise_dist='pareto', burn_in=burn_in)
    # Filter with the same (true) parameters
    V_pred_par, P_pred_par, V_filt_par, P_filt_par = kalman_like_heston_filter(y_series_pareto, **true_params, V0=1.0, P0=0.1)
    
    # 5a) Plot the results comparing Normal vs. t-distribution
    plot_heston_comparison(
        V_series_normal, V_pred_norm, V_filt_norm,
        V_series_t, V_pred_t, V_filt_t,
        dist_x="Normal",
        dist_y="t"
    )

    # 5b) Plot the results comparing Normal vs. Pareto
    plot_heston_comparison(
        V_series_normal, V_pred_norm, V_filt_norm,
        V_series_pareto, V_pred_par, V_filt_par,
        dist_x="Normal",
        dist_y="Pareto"
    )