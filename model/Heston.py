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
    # Define simulation parameters
    trading_days = 252         # Number of trading days in a year
    intraday_intervals = 39    # Number of 10-min intervals in one trading day (6.5 trading hours)
    N = trading_days * intraday_intervals + 1  # Total number of time steps
    T = 1.0  # Total time in years

    # Instantiate the Heston model with default parameters (or adjust as desired)
    heston = Heston(mu=0.1, rho=-0.5, xi=0.2, theta=0.1, kappa=0.1)
    
    # Simulate paths for asset price (S) and variance (V)
    S, V = heston.path(S0=1, V0=0.01, N=N, T=T, seed=42)
    
    # -----------------------------------------------------------------------------
    # 1. Latent True Volatility Path: Compute the square root of the variance V.
    # -----------------------------------------------------------------------------
    latent_vol = np.sqrt(V)
    
    # -----------------------------------------------------------------------------
    # 2. Return Path: Compute intraday (10-min) log returns and aggregate to daily returns.
    # -----------------------------------------------------------------------------
    log_prices = np.log(S)
    intraday_returns = np.diff(log_prices)  # 10-min log returns
    
    # Initialize lists to store daily log returns and daily realized volatility.
    daily_returns = []
    daily_rv = []  # Realized volatility for each day (computed from intraday returns)
    
    # Loop over each trading day to aggregate returns from 10-min intervals.
    for day in range(trading_days):
        # Determine the indices for the current day
        start_idx = day * intraday_intervals
        end_idx = (day + 1) * intraday_intervals
        
        # Extract the 10-min returns for the day
        day_returns = intraday_returns[start_idx:end_idx]
        
        # Daily log return is the sum of the intraday (10-min) log returns.
        daily_return = np.sum(day_returns)
        daily_returns.append(daily_return)
        
        # -----------------------------------------------------------------------------
        # 3. Realized Volatility (RV) Path:
        # Compute the realized volatility as the square root of the sum of squared intraday returns.
        # This is analogous to how you compute RV from high-frequency data.
        # -----------------------------------------------------------------------------
        day_realized_var = np.sum(day_returns**2)
        daily_rv.append(np.sqrt(day_realized_var))
    
    daily_returns = np.array(daily_returns)
    daily_rv = np.array(daily_rv)
    
    # ----------------------------
    # Plotting the Results
    # ----------------------------
    # Create a time axis for the intraday simulation and daily aggregates.
    time_intraday = np.linspace(0, T, N)        # Intraday time axis (year scale)
    time_daily = np.linspace(0, T, trading_days)  # Daily time axis (year scale)

    plt.figure(figsize=(12, 10))
    
    # Plot latent volatility path
    plt.subplot(3, 1, 1)
    plt.plot(time_intraday, latent_vol, label="Latent Volatility (√V)")
    plt.title("Latent True Volatility Path (10-min Intervals)")
    plt.xlabel("Time (years)")
    plt.ylabel("Volatility")
    plt.legend()
    
    # Plot aggregated daily returns
    plt.subplot(3, 1, 2)
    plt.plot(time_daily, daily_returns, marker='o', linestyle='-', label="Daily Log Return")
    plt.title("Daily Return Path (Aggregated from 10-min Returns)")
    plt.xlabel("Time (years)")
    plt.ylabel("Daily Log Return")
    plt.legend()
    
    # Plot daily realized volatility path
    plt.subplot(3, 1, 3)
    plt.plot(time_daily, daily_rv, marker='o', linestyle='-', color='orange', label="Daily Realized Volatility")
    plt.title("Daily Realized Volatility (RV) Path")
    plt.xlabel("Time (years)")
    plt.ylabel("Realized Volatility")
    plt.legend()
    
    plt.tight_layout()
    plt.show()