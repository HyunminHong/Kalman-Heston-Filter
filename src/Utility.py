import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class MeasurementType(Enum):
    RETURNS = "returns"
    RV = "realized_variance"
    BOTH = "both"

class Utility:
    def plot_filter_result(
        plot_index,
        titles,
        labels,
        train_filtered_list,
        test_filtered_list,
        daily_true_V,
        daily_RV,
        daily_returns,
        split_index,
        burnin=252,
        save_path=None
    ):
        """
        Plots a single Kalman filter result with RMSEs.

        Args:
            plot_index (int): Index of the model to plot.
            titles (list of str): Plot titles.
            labels (list of str): Plot labels.
            train_filtered_list (list of np.ndarray): In-sample filtered variances.
            test_filtered_list (list of np.ndarray): Out-of-sample filtered variances.
            daily_true_V (np.ndarray): True integrated variance.
            daily_RV (np.ndarray): Realized variance.
            daily_returns (np.ndarray): Used for time index.
            split_index (int): Index separating training and test sets.
            burnin (int, optional): Days to skip from start. Default is 252.
            save_path (str, optional): Path to save plot. If None, plot is shown.
        Returns:
            tuple: (rmse_in_sample, rmse_out_sample)
        """
        time_daily = np.arange(len(daily_returns)) / 252

        filt_train = train_filtered_list[plot_index][burnin:]
        true_train = daily_true_V[burnin:split_index]
        filt_test = test_filtered_list[plot_index]
        true_test = daily_true_V[split_index:]

        assert len(filt_train) == len(true_train), f"In-sample length mismatch for {titles[plot_index]}"
        assert len(filt_test) == len(true_test), f"Out-of-sample length mismatch for {titles[plot_index]}"

        rmse_is = np.sqrt(np.mean((filt_train - true_train) ** 2))
        rmse_oos = np.sqrt(np.mean((filt_test - true_test) ** 2))

        fig, ax = plt.subplots(figsize=(18, 6))

        ax.plot(time_daily[burnin:], daily_true_V[burnin:], label="True Integrated Variance", lw=2)
        ax.plot(time_daily[burnin:split_index], filt_train, label=f"{labels[plot_index]} - Train", lw=1.5)
        ax.plot(time_daily[split_index:], filt_test, label=f"{labels[plot_index]} - Test", lw=1.5, linestyle="--")
        ax.plot(time_daily[burnin:], daily_RV[burnin:], label="Realized Volatility", lw=0.4, linestyle=":")

        ax.axvline(time_daily[split_index], color='black', linestyle='--', lw=1)
        ax.text(time_daily[split_index] + 0.1, ax.get_ylim()[1]*0.95, 'Train/Test Split', color='black')

        ax.set_title(f"Kalman Filter: {titles[plot_index]}")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Variance Level")
        ax.legend(loc="upper right")
        ax.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()

        print(f"In-Sample RMSE ({titles[plot_index]}): {rmse_is:.6f}")
        print(f"Out-of-Sample RMSE ({titles[plot_index]}): {rmse_oos:.6f}")

        return rmse_is, rmse_oos
    
    def plot_all_filters(
        titles,
        labels,
        train_filtered_list,
        test_filtered_list,
        daily_true_V,
        daily_RV,
        daily_returns,
        split_index,
        burnin=252,
        save_path=None,
        figsize=(18, 6)
    ):
        """
        Plots all Kalman filters in subplots.
    
        Args:
            titles (list): Titles for each filter.
            labels (list): Labels for legend.
            train_filtered_list (list of np.ndarray): Filtered variances (train).
            test_filtered_list (list of np.ndarray): Filtered variances (test).
            daily_true_V (np.ndarray): True variance.
            daily_RV (np.ndarray): Realized variance.
            daily_returns (np.ndarray): For time axis.
            split_index (int): Index for train/test split.
            burnin (int): Burn-in period.
            figsize (tuple): Figure size.
        """
        time_daily = np.arange(len(daily_returns)) / 252
        n_filters = len(train_filtered_list)
        fig, axes = plt.subplots(n_filters, 1, figsize=(figsize[0], figsize[1] * n_filters))
    
        if n_filters == 1:
            axes = [axes]  # ensure iterable
    
        for i, ax in enumerate(axes):
            # Slice data
            filt_train = train_filtered_list[i][burnin:]
            true_train = daily_true_V[burnin:split_index]
            filt_test = test_filtered_list[i]
            true_test = daily_true_V[split_index:]
    
            # Sanity checks
            assert len(filt_train) == len(true_train), f"In-sample length mismatch for {titles[i]}"
            assert len(filt_test) == len(true_test), f"Out-of-sample length mismatch for {titles[i]}"
    
            # Plot
            ax.plot(time_daily[burnin:], daily_true_V[burnin:], label="True Integrated Variance", lw=2)
            ax.plot(time_daily[burnin:split_index], filt_train, label=f"{labels[i]} - Train", lw=1.5)
            ax.plot(time_daily[split_index:], filt_test, label=f"{labels[i]} - Test", lw=1.5, linestyle="--")
            ax.plot(time_daily[burnin:], daily_RV[burnin:], label="Realized Volatility", lw=0.4, linestyle=":")
    
            ax.axvline(time_daily[split_index], color='black', linestyle='--', lw=1)
            ax.text(time_daily[split_index] + 0.1, ax.get_ylim()[1]*0.95, 'Train/Test Split', color='black')
    
            ax.set_title(f"{titles[i]}")
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Variance Level")
            ax.legend(loc="upper right")
            ax.grid(True)
    
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()
