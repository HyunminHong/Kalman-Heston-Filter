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
            dX = (self.mu - 0.5*V[i])*dt + np.sqrt((1.0 - self.rho**2)*V[i])*dWs + self.rho*np.sqrt(V[i])*dWv
            
            X[i+1] = X[i] + dX

        S = np.exp(X)
        return S, V
    
    def path_simulation(self, S0, V0, T_years=10, trading_days=252, intraday_intervals=39, seed=None):
        # intraday_intervals is defined based on 10-min intervals per trading day (6.5 hours)
        if seed is not None: 
            np.random.seed(seed)

        N = int(T_years * trading_days * intraday_intervals)
        _, dt = np.linspace(0, T_years, N, retstep=True)

        S_high, V_high = self.path(S0, V0, N, T_years)

        # Reshape to daily matrices
        S_intraday = S_high.reshape(-1, intraday_intervals)   # shape: (2520, 39)
        V_intraday = V_high.reshape(-1, intraday_intervals)   # shape: (2520, 39)

        S_daily = S_intraday[:, -1]  # use end-of-day prices
        daily_returns = np.diff(np.log(S_daily))

        daily_true_V = V_intraday.sum(axis=1) * dt

        log_returns_intraday = np.log(S_intraday[:, 1:] / S_intraday[:, :-1])
        daily_RV = (log_returns_intraday ** 2).sum(axis=1)


        # Slice everything from index 1 onward to align with returns
        S_daily = S_daily[1:]
        daily_true_V = daily_true_V[1:]
        daily_RV = daily_RV[1:]
        time_daily = np.linspace(0, T_years, S_daily.shape[0]+1)[1:]

        return time_daily, S_daily, daily_returns, daily_true_V, daily_RV

if __name__ == "__main__":    
    # Set initial conditions and simulation parameters
    S0 = 100
    V0 = 0.04
    params = {
        'mu': 0.05,
        'rho': -0.8,
        'kappa': 2.0,
        'theta': 0.04,
        'xi': 0.2
    }
    std_asy = np.sqrt(params['theta'] * params['xi']**2 / (2 * params['kappa']))
    assert 2 * params['kappa'] * params['theta'] > params['xi']**2, "Feller condition is violated!"

    Hest = Heston(**params)
    time_daily, S_daily, daily_returns, daily_true_V, daily_RV = Hest.path_simulation(
        S0, V0, T_years=20, trading_days=252, intraday_intervals=39, seed=1
    )
    R_daily = daily_returns.copy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Subplot for BOTH measurements
    axes[0,0].plot(time_daily, S_daily, label="Daily Stock Price", lw=0.8)
    axes[0,0].set_xlabel("Time (years)")
    axes[0,0].set_ylabel("Price Level")
    axes[0,0].legend(loc="upper right")

    # Subplot for RETURNS measurement
    axes[0,1].plot(time_daily, daily_returns, label="Daily Returns", lw=0.8)
    axes[0,1].set_xlabel("Time (years)")
    axes[0,1].legend(loc="upper right")

    # Subplot for RV measurement
    axes[1,0].plot(time_daily, daily_true_V, label="True Integrated Variance", lw=0.8)
    axes[1,0].set_xlabel("Time (years)")
    axes[1,0].legend(loc="upper right")

    # Subplot for RV measurement
    axes[1,1].plot(time_daily, daily_RV, label="Daily Realized Variance", lw=0.8)
    axes[1,1].set_xlabel("Time (years)")
    axes[1,1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()