# Kalman-Heston-Filter
Approximate the Heston Model via Kalman-type filtering, assuming linear and Gaussian noise 

TODO: 
* Make different arguments for return and volatility in the Heston model (give different distributional shock)
* Note that thicker tail shock is not a driver of the volatility clustering, $\beta$ is the driver of the volatility clustering
* Find the better measure (than a simple deviation) of the fitted values
* Make a prediction for three different distributions
* Try unscented Kalman filter for t- and Pareto distribution and measure the performance (first do in-sample, then evolve to out-sample)

![output](https://github.com/user-attachments/assets/3409b1de-a275-4e76-bef0-b2be1077b0ca)
One-step forecat of Kalman-like Heston (ML) vs. GARCH (ML) under normal noise DGP.


![Heston_Kalman_sim](https://github.com/user-attachments/assets/ccbdd797-1ca3-4d81-b753-82852c9edbf8)
Kalman approximation of the Heston model with normal noise.

![normal_vs_pareto](https://github.com/user-attachments/assets/2fd24817-f592-4b1c-a938-b67a2e7ea999)
Kalman approxmiation (assuming true parameters are known) of the Heston model with Pareto ($\alpha = 3$) noise.

![normal_vs_t](https://github.com/user-attachments/assets/389beaaa-69a8-4fdd-88b5-4b2f9af5d215)
Kalman approxmiation (assuming true parameters are known) of the Heston model with Pareto ($df = 5$) noise.


