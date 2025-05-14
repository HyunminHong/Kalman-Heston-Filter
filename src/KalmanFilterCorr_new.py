import numpy as np
from scipy.optimize import minimize
from typing import Dict, Optional, List, Any
from src.Utility import MeasurementType

class HestonKalmanFilterCorr:
    """
    Kalman Filter implementation for Heston model with flexible measurement equations.
    
    This class can handle three different cases for measurement equations:
    1. One-dimensional: Returns only
    2. One-dimensional: Realized Variance (RV) only
    3. Two-dimensional: Both Returns and Realized Variance
    
    The state equation (variance process) remains consistent across all cases.
    """
    
    def __init__(self, measurement_type: MeasurementType, dt: float, V0: Optional[float] = None, P0: Optional[float] = None):
        """
        Initialize the Heston Kalman Filter.
        
        Parameters:
        measurement_type : MeasurementType
            Type of measurement data (RETURNS, RV, or BOTH).
        dt : float
            Time interval.
        V0 : float, optional
            Initial variance estimate. If None, it will be set during fitting.
        P0 : float, optional
            Initial estimation error variance. If None, it will be set during fitting.
        """
        self.measurement_type = measurement_type
        self.dt = dt
        self.V0 = V0
        self.P0 = P0
        self.params_dict = None
        self.fit_result = None
        self.burnin = 0  # default burnin is 0 (no burnin)
        
        # Data storage
        self.returns = None
        self.rv = None
        
    def set_data(self, returns: Optional[np.ndarray] = None, rv: Optional[np.ndarray] = None) -> None:
        """
        Set the measurement data for filtering.
        
        Parameters:
        returns : numpy.ndarray, optional
            Returns data, shape (T,).
        rv : numpy.ndarray, optional
            Realized variance data, shape (T,).
        """
        if self.measurement_type == MeasurementType.RETURNS and returns is None:
            raise ValueError("Returns data is required for RETURNS measurement type.")
        if self.measurement_type == MeasurementType.BOTH and (returns is None or rv is None):
            raise ValueError("Both returns and RV data are required for BOTH measurement type.")
        
        self.returns = returns
        self.rv = rv
        
        # Set initial V0 and P0 if not specified
        if self.V0 is None:
            if self.rv is not None:
                self.V0 = np.mean(self.rv)
            elif self.returns is not None:
                self.V0 = np.var(self.returns) / self.dt
            else:
                self.V0 = 0.05  # Default value
        
        if self.P0 is None:
            self.P0 = self.V0  # Reasonable default
    
    def _get_param_dict(self, params: np.ndarray) -> Dict[str, float]:
        """
        Convert parameter array to dictionary based on measurement type.
        
        Parameters:
        params : numpy.ndarray
            Parameter array with different meanings based on measurement type.
            
        Returns:
        Dict[str, float] : Parameter dictionary with named keys.
        """
        # Standard parameters for all measurement types
        param_dict = {
            'kappa': params[0],
            'theta': params[1],
            'xi': params[2]
        }
        
        # Additional parameters based on measurement type
        if self.measurement_type == MeasurementType.RETURNS:
            param_dict['mu'] = params[3]
            param_dict['rho'] = params[4]
        else:  # MeasurementType.BOTH
            param_dict['mu'] = params[3]
            param_dict['sigma'] = params[4]
            param_dict['rho'] = params[5]
            
        return param_dict
        
    def _get_measurement_matrices(self, param_dict: Dict[str, float]) -> tuple:
        """
        Get measurement model matrices based on measurement type.
        
        Parameters:
        param_dict : Dict[str, float]
            Model parameters as a dictionary.
            
        Returns:
        Tuple containing measurement model matrices:
        (mu_vec, beta_vec, sigma_vec)
        """
        if self.measurement_type == MeasurementType.RETURNS:
            # For returns only: mu = mu*dt, beta = -0.5*dt, sigma = sqrt(dt)
            mu_vec = np.array([[param_dict['mu'] * self.dt]])
            beta_vec = np.array([[-0.5 * self.dt]])
            sigma_vec = np.array([[np.sqrt(self.dt)]])
            rho_vec = np.array([[param_dict['rho']]])
        else:  # MeasurementType.BOTH
            # For both returns and RV
            mu_vec = np.array([[param_dict['mu'] * self.dt], [0]])
            beta_vec = np.array([[-0.5 * self.dt], [self.dt]])
            sigma_vec = np.array([[np.sqrt(self.dt)], [param_dict['sigma'] * np.sqrt(self.dt)]])
            rho_vec = np.array([[param_dict['rho']], [0]])
            
        return mu_vec, beta_vec, sigma_vec, rho_vec
    
    def _get_y_data(self) -> np.ndarray:
        """
        Get measurement data based on measurement type.
        
        Returns:
        y : numpy.ndarray
            Measurement data, shape (T,) for one-dimensional case or (T,2) for two-dimensional case.
        """
        if self.measurement_type == MeasurementType.RETURNS:
            return self.returns
        else:  # MeasurementType.BOTH
            return np.column_stack((self.returns, self.rv))
    
    def filter(self, params: np.ndarray, returns: Optional[np.ndarray] = None, rv: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Run Kalman filter using the specified parameters.
        
        Parameters:
        params : numpy.ndarray
            Model parameters with consistent ordering:
            For RETURNS: [kappa, theta, xi, mu]
            For RV: [kappa, theta, xi, sigma]
            For BOTH: [kappa, theta, xi, mu, sigma]
            
        Returns:
        Dict containing:
            V_filt : numpy.ndarray
                Filtered variance estimates, shape (T,).
            P_filt : numpy.ndarray
                Filtered state covariance, shape (T,).
            V_pred : numpy.ndarray
                Predicted variance estimates, shape (T,).
            P_pred : numpy.ndarray
                Predicted state covariance, shape (T,).
        """
        # Convert params to dictionary for easier access
        param_dict = self._get_param_dict(params)
        
        # Get measurement data
        if returns is not None or rv is not None:
            if self.measurement_type == MeasurementType.RETURNS:
                if returns is None:
                    raise ValueError("Returns data is required.")
                y_data = returns
            else:
                if returns is None or rv is None:
                    raise ValueError("Both returns and RV data are required.")
                y_data = np.column_stack((returns, rv))
        else:
            y_data = self._get_y_data()
            
        y = y_data
        T = y.shape[0]
        
        # Get measurement model matrices
        mu_vec, beta_vec, sigma_vec, rho_vec = self._get_measurement_matrices(param_dict)
        sigma_rho = sigma_vec * rho_vec
        
        # Extract state equation parameters
        kappa, theta, xi = param_dict['kappa'], param_dict['theta'], param_dict['xi']
        
        # Pre-allocate arrays
        V_pred = np.zeros(T)  # Predicted variance estimates
        P_pred = np.zeros(T)  # Predicted state covariance
        V_filt = np.zeros(T)  # Filtered variance estimates
        P_filt = np.zeros(T)  # Filtered state covariance
        
        # Initial conditions
        V_filt_prev = self.V0
        P_filt_prev = self.P0
        
        for t in range(T):
            ### Prediction Step & Measurement Update Step
            if self.measurement_type == MeasurementType.BOTH:
                if t == 0:
                    V_corr_pred = 0
                    P_corr_pred = 0
                else:
                    P_pred_prev = P_pred[t-1]

                    H = (beta_vec @ beta_vec.T) * P_pred_prev + (sigma_vec @ sigma_vec.T) * V_filt_prev
                    H_inv = np.linalg.pinv(H) if np.linalg.cond(H) > 1e10 else np.linalg.inv(H)
                    K_t = np.sqrt(V_filt_prev) * (sigma_rho.T @ H_inv) # in R(1x2)

                    # previous innovation
                    innov_pred = y[t-1].reshape(-1,1) - (mu_vec + beta_vec * V_pred[t-1]) # in R(2x1)


                    # correlation‐driven correction
                    V_corr_pred = xi * np.sqrt(V_filt_prev * self.dt) * (K_t @ innov_pred).item()
                    P_corr_pred = 1.0 - np.sqrt(V_filt[t-1]) * (K_t @ sigma_rho)

                # base Euler
                V_pred[t] = V_filt_prev + kappa * (theta - V_filt_prev) * self.dt + V_corr_pred

                # State noise variance Q, scaled with dt
                Q = xi**2 * V_filt_prev * self.dt
                
                # Propagate variance
                P_pred[t] = (1 - kappa * self.dt)**2 * P_filt_prev + (Q * P_corr_pred)

                # Two-dimensional case
                # Predicted measurement
                y_pred = mu_vec + beta_vec * V_pred[t]
                
                # Measurement residual (innovation)
                innovation = y[t].reshape(-1, 1) - y_pred
                
                # Measurement noise covariance using previous filtered variance
                R_mat = (V_filt_prev * (sigma_vec @ sigma_vec.T)) + (1e-8 * np.eye(2))
                R_mat = (R_mat + R_mat.T) / 2  # Ensure symmetry
                
                # Innovation covariance: S = beta*P_pred[t]*beta^T + R_mat
                S = P_pred[t] * (beta_vec @ beta_vec.T) + R_mat
                S = (S + S.T) / 2  # Ensure symmetry
                
                # Kalman gain: K = P_pred[t]*beta^T*inv(S)
                K = P_pred[t] * beta_vec.T @ np.linalg.inv(S)
                
                # Update state estimate with innovation
                V_filt[t] = V_pred[t] + (K @ innovation).item()
                
                # Update state covariance
                P_filt[t] = (1 - (K @ beta_vec).item()) * P_pred[t]
                
            else:
                # One-dimensional case for either RETURNS or RV
                # Extract scalar quantities from vectors for computational efficiency
                beta_val = beta_vec.item()
                mu_val = mu_vec.item()
                sigma_val = sigma_vec.item()
                
                # Measurement noise variance (R) based on previous filtered state
                if self.measurement_type == MeasurementType.RETURNS:
                    if t == 0:
                        V_corr_pred = 0
                        P_corr_pred = 0
                    else:
                        P_pred_prev = P_pred[t-1]

                        H = (beta_val ** 2) * P_pred_prev + (sigma_val ** 2) * V_filt_prev
                        H_inv = 0.0 if H < 1e-12 else 1 / H

                        K_t = np.sqrt(V_filt_prev) * (sigma_rho.item() * H_inv) # in R(1x1)

                        # previous innovation
                        innov_pred = y[t-1] - mu_val - beta_val * V_pred[t-1] # in R(1x1)

                        # correlation‐driven correction
                        V_corr_pred = xi * np.sqrt(V_filt_prev * self.dt) * (K_t * innov_pred)
                        P_corr_pred = 1.0 - np.sqrt(V_filt[t-1]) * (K_t * sigma_rho.item())

                    # base Euler
                    V_pred[t] = V_filt_prev + kappa * (theta - V_filt_prev) * self.dt + V_corr_pred

                    # State noise variance Q, scaled with dt
                    Q = xi**2 * V_filt_prev * self.dt
                    
                    # Propagate variance
                    P_pred[t] = (1 - kappa * self.dt)**2 * P_filt_prev + (Q * P_corr_pred)

                    R_t = V_filt_prev * self.dt  # Variance of returns
                
                # Innovation (measurement residual)
                y_pred = mu_val + beta_val * V_pred[t]
                innovation = y[t] - y_pred
                
                # Innovation variance: S = beta^2 * P_pred[t] + R_t
                S = beta_val**2 * P_pred[t] + R_t
                
                # Kalman gain: K = P_pred[t] * beta / S
                K = 0.0 if S < 1e-12 else P_pred[t] * beta_val / S
                
                # Update state estimate
                V_filt[t] = V_pred[t] + K * innovation
                
                # Update state covariance
                P_filt[t] = (1 - K * beta_val) * P_pred[t]
            
            # Ensure non-negative variance
            V_filt[t] = max(V_filt[t], 1e-6)
            
            # Update for next iteration
            V_filt_prev = V_filt[t]
            P_filt_prev = P_filt[t]
        
        return {'V_filt': V_filt, 'P_filt': P_filt, 'V_pred': V_pred, 'P_pred': P_pred}
    
    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for the Heston model. Only observations from index `burnin`
        onward are used in the summation.
        
        Parameters:
        params : numpy.ndarray
            Model parameters with consistent ordering.
            
        Returns:
        ll : float
            Log-likelihood.
        """
        # Parameter validation
        param_dict = self._get_param_dict(params)
        
        # Basic parameter bounds checking
        if param_dict['kappa'] <= 0 or param_dict['theta'] <= 0 or param_dict['xi'] <= 0:
            return -np.inf
        
        # Run the Kalman filter on the full data.
        filter_result = self.filter(params)
        V_pred = filter_result['V_pred']
        V_filt = filter_result['V_filt']
        
        # Get measurement data and model matrices
        y = self._get_y_data()
        mu_vec, beta_vec, sigma_vec, rho_vec = self._get_measurement_matrices(param_dict)
        
        T = y.shape[0]
        ll = 0.0
        
        # Use burnin: if burnin is set (>0) then start from that index,
        # otherwise default to 1 (to skip the t=0 initialization)
        start_idx = self.burnin if self.burnin > 0 else 1
        
        for t in range(start_idx, T):
            # Use previous filtered variance for consistency
            V_filt_prev = V_filt[t-1]
            
            if self.measurement_type == MeasurementType.BOTH:
                # Two-dimensional case
                # Predicted measurement
                y_pred = mu_vec + beta_vec * V_pred[t]
                
                # Measurement noise covariance using previous filtered variance
                R_mat = (V_filt_prev * (sigma_vec @ sigma_vec.T)) + (1e-8 * np.eye(2))
                R_mat = (R_mat + R_mat.T) / 2  # Ensure symmetry
                
                # Innovation
                innovation = y[t].reshape(-1, 1) - y_pred
                
                # Innovation covariance: S = beta*P_pred[t]*beta^T + R_mat
                beta_squared = beta_vec @ beta_vec.T
                S = filter_result['P_pred'][t] * beta_squared + R_mat
                S = (S + S.T) / 2  # Ensure symmetry
                
                try:
                    det_S = np.linalg.det(S)
                    if det_S <= 0:
                        return -np.inf
                    
                    inv_S = np.linalg.inv(S)
                    ll_t = -np.log(2 * np.pi) - 0.5 * np.log(det_S) - 0.5 * (innovation.T @ inv_S @ innovation).item()
                except:
                    return -np.inf
                
            else:
                # One-dimensional case
                beta_val = beta_vec.item()
                mu_val = mu_vec.item()
                
                # Measurement noise variance based on measurement type using previous filtered variance
                if self.measurement_type == MeasurementType.RETURNS:
                    R_t = V_filt_prev * self.dt  # Variance of returns
                
                # Innovation variance: S = beta^2 * P_pred[t] + R_t
                S = beta_val**2 * filter_result['P_pred'][t] + R_t
                
                if S <= 0:
                    return -np.inf
                
                # Innovation
                y_pred = mu_val + beta_val * V_pred[t]
                innovation = y[t] - y_pred
                
                ll_t = -0.5 * np.log(2.0 * np.pi * S) - 0.5 * (innovation**2 / S)
            
            ll += ll_t
        
        return ll
    
    def negative_log_likelihood(self, params: np.ndarray) -> float:
        """
        Negative log-likelihood function for optimization.
        
        Parameters:
        params : numpy.ndarray
            Model parameters.
            
        Returns:
        nll : float
            Negative log-likelihood.
        """
        ll = self.log_likelihood(params)
        return -ll
    
    def fit(self, returns: Optional[np.ndarray] = None, rv: Optional[np.ndarray] = None, 
            initial_params: Optional[np.ndarray] = None, bounds: Optional[List] = None, 
            optimizer_kwargs: Optional[Dict[str, Any]] = None, burnin: int = 0) -> Dict[str, Any]:
        """
        Estimate model parameters using maximum likelihood.
        
        Parameters:
        returns : numpy.ndarray, optional
            Returns data, shape (T,). Required if measurement_type includes returns.
        rv : numpy.ndarray, optional
            Realized variance data, shape (T,). Required if measurement_type includes RV.
        initial_params : numpy.ndarray, optional
            Initial parameter values with consistent ordering:
            For RETURNS: [kappa, theta, xi, mu]
            For RV: [kappa, theta, xi, sigma]
            For BOTH: [kappa, theta, xi, mu, sigma]
        bounds : list, optional
            Parameter bounds for optimization.
        optimizer_kwargs : dict, optional
            Additional keyword arguments for the optimizer.
        burnin : int, optional
            Number of initial observations to drop in the likelihood calculation.
            
        Returns:
        Dict containing optimization results.
        """
        # Set burnin for likelihood computation
        self.burnin = burnin
        
        # Set data if provided
        if returns is not None or rv is not None:
            self.set_data(returns, rv)
        
        # Check if data is set
        if self.measurement_type == MeasurementType.RETURNS and self.returns is None:
            raise ValueError("Returns data is required for RETURNS measurement type.")
        if self.measurement_type == MeasurementType.BOTH and (self.returns is None or self.rv is None):
            raise ValueError("Both returns and RV data are required for BOTH measurement type.")
        
        # Set default initial parameters if not provided
        if initial_params is None:
            # Common parameters
            kappa_init = 5.0
            theta_init = np.var(self.returns) / self.dt if self.measurement_type == MeasurementType.RETURNS else np.mean(self.rv)
            xi_init = 0.5
            
            if self.measurement_type == MeasurementType.RETURNS:
                mu_init = np.mean(self.returns) / self.dt
                rho_init = 0
                initial_params = np.array([kappa_init, theta_init, xi_init, mu_init, rho_init])
            else:  # MeasurementType.BOTH
                mu_init = np.mean(self.returns) / self.dt
                sigma_init = 0.5
                rho_init = 0
                initial_params = np.array([kappa_init, theta_init, xi_init, mu_init, sigma_init, rho_init])
        
        if bounds is None:
            kappa_bounds = (1e-6, None)
            theta_bounds = (1e-6, None)
            xi_bounds = (1e-6, None)
            
            if self.measurement_type == MeasurementType.RETURNS:
                mu_bounds = (-0.2, 0.2)
                rho_bounds = (-0.99, 0.99)
                bounds = [kappa_bounds, theta_bounds, xi_bounds, mu_bounds, rho_bounds]
            else:  # MeasurementType.BOTH
                mu_bounds = (-0.2, 0.2)
                sigma_bounds = (1e-6, 2.0)
                rho_bounds = (-0.99, 0.99)
                bounds = [kappa_bounds, theta_bounds, xi_bounds, mu_bounds, sigma_bounds, rho_bounds]
        
        default_optimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': bounds,
            'options': {'disp': True}
        }
        
        if optimizer_kwargs:
            default_optimizer_kwargs.update(optimizer_kwargs)
        
        result = minimize(
            fun=self.negative_log_likelihood,
            x0=initial_params,
            **default_optimizer_kwargs
        )
        
        self.params_dict = self._get_param_dict(result.x)
        self.fit_result = result
        
        print("\nQMLE Results for Heston Model:")
        print("-" * 50)
        for name, value in self.params_dict.items():
            print(f"{name}: {value:.6f}")
        print(f"Negative Log-Likelihood: {result.fun:.6f}")
        print(f"Convergence: {result.success}")
        if not result.success:
            print(f"Message: {result.message}")
        print("-" * 50)
        
        return {
            'params': result.x,
            'params_dict': self.params_dict,
            'nll': result.fun,
            'success': result.success,
            'message': result.message,
            'optimization_result': result
        }
    
    def get_filtered_variance(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get filtered variance estimates.
        
        Parameters:
        params : numpy.ndarray, optional
            Model parameters. If None, use fitted parameters.
            
        Returns:
        V_filt : numpy.ndarray
            Filtered variance estimates, shape (T,).
        """
        if params is None:
            if self.params_dict is None:
                raise ValueError("No parameters available. Call fit() first or provide parameters.")
            if self.measurement_type == MeasurementType.RETURNS:
                params = np.array([
                    self.params_dict['kappa'], 
                    self.params_dict['theta'], 
                    self.params_dict['xi'], 
                    self.params_dict['mu'],
                    self.params_dict['rho']
                ])
            else:  # MeasurementType.BOTH
                params = np.array([
                    self.params_dict['kappa'], 
                    self.params_dict['theta'], 
                    self.params_dict['xi'], 
                    self.params_dict['mu'], 
                    self.params_dict['sigma'],
                    self.params_dict['rho']
                ])
        
        filter_result = self.filter(params)
        return filter_result['V_filt']
    
    def summary(self) -> None:
        """Print a summary of the model and fitted parameters."""
        if self.params_dict is None:
            print("Model not fitted yet. Call fit() first.")
            return
        
        print("\nHeston Kalman Filter Summary")
        print("=" * 50)
        print(f"Measurement Type: {self.measurement_type.value}")
        print(f"Time Interval (dt): {self.dt}")
        print(f"Initial Variance (V0): {self.V0:.6f}")
        print(f"Initial Covariance (P0): {self.P0:.6f}")
        
        print("\nFitted Parameters:")
        print("-" * 50)
        
        for name, value in self.params_dict.items():
            print(f"{name}: {value:.6f}")
        
        print(f"\nNegative Log-Likelihood: {self.fit_result.fun:.6f}")
        print(f"Convergence: {self.fit_result.success}")
        if not self.fit_result.success:
            print(f"Message: {self.fit_result.message}")
        
        # Calculate AIC and BIC
        n_params = len(self.params_dict)
        T = len(self._get_y_data())
        
        aic = 2 * self.fit_result.fun + 2 * n_params
        bic = 2 * self.fit_result.fun + n_params * np.log(T)
        
        print(f"\nAIC: {aic:.6f}")
        print(f"BIC: {bic:.6f}")
        print("=" * 50)