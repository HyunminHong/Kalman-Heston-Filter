import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from model.kalman_heston import kalman_like_heston_filter, estimate_params_qmle
from model.heston_simulation import simulate_heston
from model.draw_noise import draw_noise

def plot_heston_comparison(
    V_true_norm, V_pred_norm, V_filt_norm,
    V_true_t, V_pred_t, V_filt_t,
    title="Comparison: Normal vs. t-distributed Heston"
):
    """
    Create a 2x2 figure to compare the Heston simulation+filter results
    under normal vs. t-distributed noise.

    Top row: Normal noise
       - Left: True vs. predicted vs. filtered
       - Right: Error (True - Filter)
    Bottom row: t-distributed noise
       - Left: True vs. predicted vs. filtered
       - Right: Error (True - Filter)
    """
    # Determine lengths (assuming both have the same T, but not required)
    T_norm = len(V_filt_norm)
    T_t = len(V_filt_t)
    
    # Create the figure
    fig, axs = plt.subplots(2, 2, figsize=(14, 8))
    
    # ---- TOP ROW: Normal simulation ----
    time_norm = np.arange(T_norm)
    # Left: True vs. filter
    axs[0, 0].plot(time_norm, V_true_norm[:T_norm], label="True V (Normal)", color="blue")
    axs[0, 0].plot(time_norm, V_pred_norm, label="Kalman Pred", linestyle="--", color="red")
    axs[0, 0].plot(time_norm, V_filt_norm, label="Kalman Filter", linestyle="-.", color="green")
    axs[0, 0].set_title("Normal Noise: Heston vs. Kalman")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Variance V")
    axs[0, 0].legend()
    
    # Right: Error
    error_norm = V_true_norm[:T_norm] - V_filt_norm
    axs[0, 1].plot(time_norm, error_norm, color="purple", label="Error (True - Filter)")
    axs[0, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[0, 1].set_title("Normal Noise: Filtering Error")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].set_ylabel("Error")
    axs[0, 1].legend()
    
    # ---- BOTTOM ROW: t-distributed simulation ----
    time_t = np.arange(T_t)
    # Left: True vs. filter
    axs[1, 0].plot(time_t, V_true_t[:T_t], label="True V (t-dist)", color="blue")
    axs[1, 0].plot(time_t, V_pred_t, label="Kalman Pred", linestyle="--", color="red")
    axs[1, 0].plot(time_t, V_filt_t, label="Kalman Filter", linestyle="-.", color="green")
    axs[1, 0].set_title("t-Distributed Noise: Heston vs. Kalman")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Variance V")
    axs[1, 0].legend()
    
    # Right: Error
    error_t = V_true_t[:T_t] - V_filt_t
    axs[1, 1].plot(time_t, error_t, color="purple", label="Error (True - Filter)")
    axs[1, 1].axhline(0, color="black", linestyle="--", linewidth=1)
    axs[1, 1].set_title("t-Distributed Noise: Filtering Error")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Error")
    axs[1, 1].legend()
    
    # Overall title
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # 1) True parameters for the Heston model
    alpha_true, beta_true, gamma_true, delta_true = 0.1, 0.9, 0.3, 0.2
    T = 1000
    burn_in = 200
    seed = 42

    # 2) Simulate under Normal noise
    V_series_normal, y_series_normal = simulate_heston(
        T, alpha_true, beta_true, gamma_true, delta_true,
        V0=1.0, seed=seed, noise_dist='normal', burn_in=burn_in
    )
    # Kalman-like filter with the *true* parameters (just for illustration)
    V_pred_norm, P_pred_norm, V_filt_norm, P_filt_norm = kalman_like_heston_filter(
        y_series_normal, alpha_true, beta_true, gamma_true, delta_true,
        V0=1.0, P0=0.1
    )
    
    # 3) Simulate under t-distributed noise
    #    e.g., df=4 for a moderate heavy tail
    V_series_t, y_series_t = simulate_heston(
        T, alpha_true, beta_true, gamma_true, delta_true,
        V0=1.0, seed=seed, noise_dist='t', noise_params={'df':4},
        burn_in=burn_in
    )
    # Filter with the same (true) parameters
    V_pred_t, P_pred_t, V_filt_t, P_filt_t = kalman_like_heston_filter(
        y_series_t, alpha_true, beta_true, gamma_true, delta_true,
        V0=1.0, P0=0.1
    )
    
    # 4) Plot both results side by side in one figure
    plot_heston_comparison(
        V_series_normal, V_pred_norm, V_filt_norm,
        V_series_t, V_pred_t, V_filt_t,
        title="Comparison of Heston Simulation: Normal vs. t Noise"
    )