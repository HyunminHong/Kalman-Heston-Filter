# Kalman-Heston-Filter
Approximate the Heston Model via Kalman-like filtering, assuming linear and Gaussian noise 

TODO: 
* Check whether two-dimensional case is well implemented. It seem to have lags compared to 1-dim measurements.
* Parameter consistency? Three different measurement spaces assume technically different models. Justify this.

![simulation_filtering](https://github.com/user-attachments/assets/f273bbd4-67b8-40b2-af30-53c3feee9c45)
Heston Kalman-like filter applied to:
1. Returns and Realized Variance (RV) (2-dim measurement space)
2. Returns (1-dim measurement space)
3. Realized Variance (RV) (1-dim measurement space)

![VSP](https://github.com/user-attachments/assets/864d7a77-ca3f-4d2b-ba5e-3d90a126d6a4)
Volatility Signature Plot (SPY)

The above VSP justifies to use 10-minute realized volatility 
