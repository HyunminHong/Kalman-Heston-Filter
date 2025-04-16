# Kalman-Heston-Filter
Approximate the Heston Model via Kalman-like filtering, assuming linear and Gaussian noise 

TODO: 
* Check whether two-dimensional case is well implemented. It seem to have lags compared to 1-dim measurements.
* Parameter consistency? Three different measurement spaces assume technically different models. Justify this.

The below simulation is done in the following manner: 
1. **Compute total intraday steps**  
   - Calculate \(N = \text{T\_years} \times \text{trading\_days} \times \text{intraday\_intervals}\).  
   *Determines how many 10‑minute observations will be simulated over the entire period.*

2. **Determine time increment**  
   - Use `np.linspace(0, T_years, N, retstep=True)` to obtain `dt`.  
   *Finds the uniform time step corresponding to each 10‑minute interval.*

3. **Simulate high‑frequency paths**  
   - Call `self.path(S0, V0, N, T_years)` to generate arrays `S_high` (prices) and `V_high` (variances) of length \(N\).  
   *Produces the intraday price and variance trajectories at 10‑minute resolution.*

4. **Reshape into daily matrices**  
   - Reshape `S_high` and `V_high` into shape `(trading_days * T_years, intraday_intervals)` so each row is one trading day of 39 intervals.  
   *Organizes the 10‑minute data into a days×39 grid.*

5. **Extract end‑of‑day prices & compute daily returns**  
   - Take the last column of `S_intraday` as `S_daily`, then compute `daily_returns = diff(log(S_daily))`.  
   *Obtains one log‑return per day based on end‑of‑day prices.*

6. **Compute true daily integrated variance**  
   - Sum each row of `V_intraday` and multiply by `dt` to get `daily_true_V`.  
   *Aggregates the model’s instantaneous variances into a daily “true” IV.*

7. **Compute realized variance**  
   - Compute intraday log‑returns `log(S_intraday[:,1:]/S_intraday[:,:-1])` then sum their squares row‑wise to get `daily_RV`.  
   *Forms daily realized variance from squared 10‑minute returns.*

8. **Align series for returns**  
   - Drop the first element of `S_daily`, `daily_true_V`, and `daily_RV` so they align with the return vector (which is one element shorter).  
   *Ensures that each return has a matching variance observation.*

9. **Build daily time axis**  
   - Create `time_daily = linspace(0, T_years, len(S_daily)+1)[1:]` to span the simulation in years.  
   *Summary: Generates the corresponding time stamps for each day’s data point.*

10. **Return aggregated arrays**  
    - Output `(time_daily, S_daily, daily_returns, daily_true_V, daily_RV)`.  
    *Summary: Provides the daily time vector, end‑of‑day prices, log‑returns, true integrated variance, and realized variance for further analysis.*

![simulation_filtering](https://github.com/user-attachments/assets/f273bbd4-67b8-40b2-af30-53c3feee9c45)
Heston Kalman-like filter applied to:
1. Returns and Realized Variance (RV) (2-dim measurement space)
2. Returns (1-dim measurement space)
3. Realized Variance (RV) (1-dim measurement space)

![VSP](https://github.com/user-attachments/assets/864d7a77-ca3f-4d2b-ba5e-3d90a126d6a4)
Volatility Signature Plot (SPY)

The above VSP justifies to use 10-minute realized volatility 
