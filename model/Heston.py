import numpy as np
import matplotlib.pyplot as plt

class Heston:
    def __init__(self, mu=0.1, rho=-0.5, xi=0.2, theta=0.1, kappa=0.1):
        """
        mu      : drift of the log-price under the risk-neutral measure
        rho     : correlation between the Brownian motions driving S and V
        xi      : volatility of volatility 
        theta   : long-run mean of the variance process
        kappa   : speed of mean reversion of the variance
        """
        if abs(rho) > 1:
            raise ValueError("|rho| must be <= 1")
        if theta < 0 or xi < 0 or kappa < 0:
            raise ValueError("theta, xi, kappa must be nonnegative")
        self.mu = mu
        self.rho = rho
        self.xi = xi
        self.theta = theta
        self.kappa = kappa

    def path(self, S0, V0, N, T=1.0, seed=None):
        if seed is not None:
            np.random.seed(seed)

        dt = T / (N - 1)  # time increment
        
        # Draw i.i.d. standard normals for W^S and W^V
        Zs = np.random.normal(0, 1, size=N-1)
        Zv = np.random.normal(0, 1, size=N-1)

        # Initialize arrays for log-price X and variance V
        X = np.zeros(N)
        V = np.zeros(N)
        X[0] = np.log(S0)
        V[0] = V0

        for i in range(N-1):
            # Brownian increments (Euler-Maruyama)
            dWs = np.sqrt(dt) * Zs[i]
            dWv = np.sqrt(dt) * Zv[i]

            # Variance update (4.5b)
            V[i+1] = np.abs(V[i] + self.kappa*(self.theta - V[i])*dt + self.xi*np.sqrt(V[i])*dWv)
            
            # Log-price update (4.5a)
            # R_i = (mu - 0.5*V[i])*dt + sqrt((1-rho^2)*V[i])*dWs + rho*sqrt(V[i])*dWv
            dX = (self.mu - 0.5*V[i])*dt + np.sqrt((1.0 - self.rho**2)*V[i])*dWs + self.rho*np.sqrt(V[i])*dWv
            
            X[i+1] = X[i] + dX

        # Exponentiate to get the actual price path
        S = np.exp(X)
        return S, V
    
if __name__ == "__main__":
    # For a 5-year simulation
    trading_days = 252 * 5       # Total trading days (5 years)
    intraday_intervals = 39      # 10-min intervals in one trading day (6.5 trading hours)
    N = trading_days * intraday_intervals + 1  # Total number of time steps
    T = 5.0  # Total time in years

    dt = T / (N-1)

    heston = Heston(mu=0.1, rho=-0., xi=0.2, theta=0.1, kappa=0.1)

    # Simulate paths for asset price (S) and variance (V)
    S, V = heston.path(S0=1, V0=0.01, N=N, T=T, seed=42)

    # Compute the intraday 10-min log returns from prices
    log_prices = np.log(S)
    intraday_returns = np.diff(log_prices)  # 10-min log returns

    daily_returns = []
    daily_rv = []

    for day in range(trading_days):
        start_idx = day * intraday_intervals
        end_idx = (day + 1) * intraday_intervals
        
        # Extract the intraday returns for the day.
        day_returns = intraday_returns[start_idx:end_idx]
        
        # Daily log return: sum of intraday returns.
        daily_return = np.sum(day_returns)
        daily_returns.append(daily_return)
        
        # Daily realized variance (RV): sum of squared intraday returns.
        day_realized_var = np.sum(day_returns**2)
        daily_rv.append(day_realized_var)

    daily_returns = np.array(daily_returns)
    daily_rv = np.array(daily_rv)

    # Also, downsample the latent variance V to daily by computing the average variance for each day.
    daily_true_var = [np.mean(V[day * intraday_intervals:(day + 1) * intraday_intervals]) for day in range(trading_days)]
    daily_true_var = np.array(daily_true_var)

    days = np.arange(1, trading_days + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # First subplot: Daily Realized Variance vs. Daily Latent Variance (True V)
    axes[0].plot(days, daily_true_var, label="Daily Latent Variance (True V)", linewidth=2)
    axes[0].plot(days, daily_rv * 252, label="Daily Realized Variance (RV)", linestyle='--', color='orange')
    axes[0].set_title("Daily Latent Variance vs. Realized Variance over 5 Years")
    axes[0].set_xlabel("Trading Day")
    axes[0].set_ylabel("Variance")
    axes[0].legend()

    # Second subplot: Daily Log Returns
    axes[1].plot(days, daily_returns, marker='o', linestyle='-', color='green')
    axes[1].set_title("Daily Log Returns over 5 Years")
    axes[1].set_xlabel("Trading Day")
    axes[1].set_ylabel("Daily Log Return")

    plt.tight_layout()
    plt.show()
