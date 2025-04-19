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
        S_intraday = S_high.reshape(-1, intraday_intervals)   # shape: (2500, 39)
        V_intraday = V_high.reshape(-1, intraday_intervals)   # shape: (2500, 39)

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
    # Initializations
    S0 = 100
    V0 = 0.04
    params = {
        'mu': 0.05, 
        'rho': -0.8,
        'kappa': 1,
        'theta': 0.04,
        'xi': 0.2
    }
    
    std_asy = np.sqrt(params['theta'] * params['xi']**2 / (2 * params['kappa']))  # asymptotic standard deviation for the CIR process
    assert 2 * params['kappa'] * params['theta'] > params['xi']**2  # Feller condition

    Hest = Heston(**params)
    time_daily, S_daily, daily_true_V, daily_RV = Hest.path_simulation(S0, V0, T_years=10, trading_days=252, intraday_intervals=39, seed=10)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 4))

    ax1.plot(time_daily, S_daily)
    ax1.set_title("Daily Stock Prices (End-of-Day)")
    ax1.set_xlabel("Time (years)")
    ax1.set_ylabel("Stock Price")

    ax2.plot(time_daily[1:], np.diff(np.log(S_daily)))
    ax2.set_title("Daily Log Return")
    ax2.set_xlabel("Time (years)")
    ax2.set_ylabel("Log Return")

    ax3.plot(time_daily, daily_true_V)
    ax3.set_title("Daily True Integrated Variance")
    ax3.set_xlabel("Time (years)")
    ax3.set_ylabel("Integrated Variance")

    ax4.plot(time_daily, daily_RV)
    ax4.set_title("Daily Realized Volatility (10-min data)")
    ax4.set_xlabel("Time (years)")
    ax4.set_ylabel("Realized Volatility")

    plt.tight_layout()
    plt.show()