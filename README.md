# Kalman-Heston-Filter
Approximate the Heston Model via Kalman-type filtering, assuming linear and Gaussian noise 

TODO: 
* Make different arguments for return and volatility in the Heston model (give different distributional shock)
* Note that thicker tail shock is not a driver of the volatility clustering, $\beta$ is the driver of the volatility clustering
* Find the better measure (than a simple deviation) of the fitted values
* Make a prediction for three different distributions
* Try unscented Kalman filter for t- and Pareto distribution and measure the performance (first do in-sample, then evolve to out-sample)

The below is the Kalman approximation of the Heston model. Note that the DGP of the Heston model still assumes the normal distributed noise.
![Heston_Kalman_sim](https://github.com/user-attachments/assets/ccbdd797-1ca3-4d81-b753-82852c9edbf8)
