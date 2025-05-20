import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

class RealizedGARCH:
    """
    Implementation of the Realized GARCH model as described in:
    Hansen, P. R., Huang, Z., & Shek, H. H. (2012). Realized GARCH: a joint model for returns and realized measures of volatility.
    """
    
    def __init__(self, p=1, q=1):
        """
        Initialize the Realized GARCH model with order parameters.
        
        Parameters:
        - p: Order of GARCH terms
        - q: Order of ARCH terms
        """
        self.p = p
        self.q = q
        self.params = None
        self.param_names = None
        self.fitted = False
        
    def _compute_volatility(self, returns, realized_vol, params):
        """
        Compute conditional volatility (h_t) series based on given parameters.
        
        The Realized GARCH(p,q) model with log specification:
        
        log(h_t) = omega + sum_{i=1}^p beta_i * log(h_{t-i}) + sum_{j=1}^q gamma_j * log(RV_{t-j})
        
        For simplicity, we implement the Realized GARCH(1,1) model:
        log(h_t) = omega + beta * log(h_{t-1}) + gamma * log(RV_{t-1})
        """
        omega, beta, gamma = params[:3]
        
        T = len(returns)
        h = np.zeros(T+1)  # +1 to store initial volatility
        
        # Initialize h_1 with the empirical variance
        h[0] = np.var(returns)
        
        for t in range(1, T+1):
            if t == 1:
                # For the first iteration, use the initial value
                log_h = omega + beta * np.log(h[0]) + gamma * np.log(realized_vol[0])
            else:
                log_h = omega + beta * np.log(h[t-1]) + gamma * np.log(realized_vol[t-2])
            
            h[t] = np.exp(log_h)
            
        return h[1:]  # Remove the initial value
    
    def _measurement_equation(self, returns, realized_vol, h, params):
        """
        Implement the measurement equation:
        log(RV_t) = xi + phi * log(h_t) + tau * (z_t) + u_t
        
        where u_t is assumed to be Gaussian with mean 0 and variance sigma_u^2
        """
        xi, phi, tau_1, tau_2, sigma_u = params[3:8]
        
        T = len(realized_vol)
        log_rv_hat = np.zeros(T)
        
        for t in range(T):
            if t > 0:  # Skip the first observation
                z_t = returns[t] / np.sqrt(h[t])
                log_rv_hat[t] = xi + phi * np.log(h[t]) + tau_1 * z_t + tau_2 * (z_t**2 - 1)
        
        return log_rv_hat[1:]  # Remove the first value which can't be computed
    
    def _negative_log_likelihood(self, params, returns, realized_vol):
        """
        Compute the negative log-likelihood function for the Realized GARCH model.
        
        For Realized GARCH(1,1), the parameters are:
        - omega, beta, gamma (conditional volatility equation)
        - xi, phi, tau_1, tau_2, sigma_u (measurement equation)
        - sigma_z (return innovations)
        """
        if len(params) != 9:
            raise ValueError("Expected 9 parameters for Realized GARCH(1,1) model")
        
        omega, beta, gamma, xi, phi, tau_1, tau_2, sigma_u, sigma_z = params
        
        # Parameter constraints
        if (beta < 0 or beta > 0.999 or gamma < 0 or (beta + gamma) > 0.999 or
            sigma_u <= 0 or sigma_z <= 0):
            return 1e10  # Return a large value for invalid parameters
        
        T = len(returns)
        
        # Compute conditional volatilities
        h = self._compute_volatility(returns, realized_vol, params)
        
        # Compute log-likelihood for returns
        ll_returns = -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + (returns**2) / h)
        
        # Compute log-likelihood for realized volatility (measurement equation)
        log_rv_hat = self._measurement_equation(returns, realized_vol, h, params)  # Pass returns here
        u = np.log(realized_vol[1:]) - log_rv_hat  # Measurement equation residuals
        ll_rv = -0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma_u**2) + (u**2) / (sigma_u**2))
        
        # Total log-likelihood
        ll = ll_returns + ll_rv
        
        return -ll  # Return negative log-likelihood for minimization
    
    def fit(self, returns, realized_vol, initial_params=None):
        """
        Fit the Realized GARCH model to the data.
        
        Parameters:
        - returns: Array of return observations
        - realized_vol: Array of realized volatility observations
        - initial_params: Initial parameter values (optional)
        
        Returns:
        - Parameter estimates
        """
        # Default initial parameters if not provided
        if initial_params is None:
            # [omega, beta, gamma, xi, phi, tau_1, tau_2, sigma_u, sigma_z]
            initial_params = [0.01, 0.7, 0.2, -0.5, 0.9, 0.1, 0.05, 0.2, 1.0]
        
        # Parameter names for reference
        self.param_names = ['omega', 'beta', 'gamma', 'xi', 'phi', 
                           'tau_1', 'tau_2', 'sigma_u', 'sigma_z']
        
        # Minimize negative log-likelihood
        result = minimize(
            self._negative_log_likelihood, 
            initial_params, 
            args=(returns, realized_vol),
            method='BFGS',
            options={'disp': False}
        )
        
        self.params = result.x
        self.fitted = True
        
        # Store the fitted volatilities
        self.fitted_volatility = self._compute_volatility(returns, realized_vol, self.params)
        
        # Create a summary of the fitted parameters
        self.summary = pd.DataFrame({
            'Parameter': self.param_names,
            'Estimate': self.params
        })
        
        return self.summary
    
    def forecast_volatility(self, returns, realized_vol, h_ahead=1):
        """
        Forecast conditional volatility h periods ahead.
        
        Parameters:
        - returns: Array of return observations
        - realized_vol: Array of realized volatility observations
        - h_ahead: Number of periods to forecast ahead
        
        Returns:
        - Volatility forecasts
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Extract parameters
        omega, beta, gamma = self.params[:3]
        
        # Compute current conditional volatility
        h = self._compute_volatility(returns, realized_vol, self.params)
        current_h = h[-1]
        
        # For Realized GARCH(1,1), the h-step ahead forecast is:
        # log(h_{T+h}) = omega * (1 - (beta + gamma)^h) / (1 - beta - gamma) + (beta + gamma)^h * log(h_T)
        forecast = np.zeros(h_ahead)
        persistence = beta + gamma
        
        for h in range(1, h_ahead + 1):
            log_h_forecast = omega * (1 - persistence**h) / (1 - persistence) + persistence**h * np.log(current_h)
            forecast[h-1] = np.exp(log_h_forecast)
        
        return forecast
    
    def predict(self, returns, realized_vol):
        """
        Predict conditional volatility for the given returns and realized volatility series.
        
        Parameters:
        - returns: Array of return observations
        - realized_vol: Array of realized volatility observations
        
        Returns:
        - Predicted conditional volatilities
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predicting")
        
        h = self._compute_volatility(returns, realized_vol, self.params)
        return h
        
    def get_fitted_volatility(self, returns, realized_vol):
        """
        Get the fitted/in-sample conditional volatility values.
        
        Parameters:
        - returns: Array of return observations used for fitting
        - realized_vol: Array of realized volatility observations used for fitting
        
        Returns:
        - Fitted conditional volatilities
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before getting fitted values")
        
        h = self._compute_volatility(returns, realized_vol, self.params)
        return h

# Example usage
if __name__ == "__main__":
    # Generate some artificial data
    np.random.seed(42)
    T = 1000
    
    # Generate returns based on GARCH process
    h = np.zeros(T)
    returns = np.zeros(T)
    h[0] = 0.1
    
    # Parameters for simulation
    omega, beta, gamma = 0.05, 0.7, 0.2
    xi, phi = -0.5, 0.9
    tau_1, tau_2 = 0.1, 0.05
    
    for t in range(1, T):
        h[t] = omega + beta * h[t-1] + gamma * (returns[t-1]**2)
        returns[t] = np.sqrt(h[t]) * np.random.normal(0, 1)
    
    # Generate realized volatility with some noise
    rv = h * np.random.lognormal(0, 0.3, T)
    
    # Split into training and testing sets
    train_size = int(0.7 * T)
    train_returns = returns[:train_size]
    train_rv = rv[:train_size]
    test_returns = returns[train_size:]
    test_rv = rv[train_size:]
    
    # Fit the model
    model = RealizedGARCH(p=1, q=1)
    model.fit(train_returns, train_rv)
    
    # Get fitted volatility (in-sample)
    fitted_vol = model.get_fitted_volatility(train_returns, train_rv)
    
    # Predict volatility (out-of-sample)
    predicted_vol = model.predict(test_returns, test_rv)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot in-sample fit
    plt.subplot(2, 1, 1)
    plt.plot(np.sqrt(h[:train_size]), label='True Volatility (In-sample)')
    plt.plot(np.sqrt(fitted_vol), label='Fitted Volatility')
    plt.legend()
    plt.title('Realized GARCH In-Sample Fit')
    
    # Plot out-of-sample prediction
    plt.subplot(2, 1, 2)
    plt.plot(np.sqrt(h[train_size:]), label='True Volatility (Out-of-sample)')
    plt.plot(np.sqrt(predicted_vol), label='Predicted Volatility')
    plt.legend()
    plt.title('Realized GARCH Out-of-Sample Prediction')
    
    plt.tight_layout()
    plt.show()