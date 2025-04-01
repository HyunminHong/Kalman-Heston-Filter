import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
 
csv_path = r"C:\Users\HONGHY\Documents\3. Data\lvvzq2yrdnkigz87.csv"
 
blocksize = 100 * 1024 * 1024  # 100 MB
 
df = dd.read_csv(csv_path, blocksize=blocksize)
 
df['timestamp'] = dd.to_datetime(df['DATE'] + ' ' + df['TIME_M'])
 
df['timestamp_sec'] = df['timestamp'].dt.floor('s')
 
aggregated = df.groupby('timestamp_sec').agg({'BID': 'mean', 'ASK': 'mean'})
 
result = aggregated.compute().sort_index()
 
sec_range = range(5, 3601, 5)
 
volatility_dense = []
 
for s in sec_range:
    freq_str = f"{s}S"
    
    resampled_price = result["PRICE"].resample(freq_str).last().dropna()
    log_returns = np.log(resampled_price / resampled_price.shift(1))
   
    daily_var = log_returns.groupby(log_returns.index.date).apply(lambda x: np.sum(x**2))
    daily_rv = np.sqrt(daily_var)
   
    mean_rv = daily_rv.mean()
    volatility_dense.append(mean_rv)
 
fig, ax = plt.subplots(figsize=(10, 5))
 
ax.scatter(sec_range, volatility_dense, c='blue', marker='.', alpha=0.8, label='Daily RV')
 
ax.axhline(y=0.015, color='red', linestyle='--', linewidth=1.0)
 
ax.set_title('Volatility Signature Plot (Dense Sampling)')
ax.set_xlabel('Sampling Interval (seconds)')
ax.set_ylabel('Daily Realized Volatility')
 
tickvals = [60, 300, 600, 900, 1800, 3600]  # e.g. 1m, 5m, 10m, 15m, 30m, 60m
ticklabels = ["1m", "5m", "10m", "15m", "30m", "60m"]
ax.set_xticks(tickvals)
ax.set_xticklabels(ticklabels)
 
ax.legend()
plt.tight_layout()
plt.show()