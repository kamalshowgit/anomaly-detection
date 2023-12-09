import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points
num_points = 1000

# Time vector
time = np.arange(num_points)

# Simulate normal behavior (linear trend + sinusoidal seasonality + noise)
trend = 0.05 * time
seasonality = 5 * np.sin(2 * np.pi * time / 24)
noise = np.random.normal(scale=0.5, size=num_points)
normal_data = trend + seasonality + noise

# Introduce anomalies
num_anomalies = int(0.45 * num_points)
anomaly_indices = np.random.choice(num_points, size=num_anomalies, replace=False)
normal_data[anomaly_indices] += 5  # Add a significant value to create anomalies

# Ensure positive values
normal_data = np.abs(normal_data)

# Create dataframe
data = {'Time': time, 'Metric': normal_data, 'Anomaly': 0}
df = pd.DataFrame(data)

# Mark anomalies in the 'Anomaly' column
df.loc[anomaly_indices, 'Anomaly'] = 1

# Export to CSV
df.to_csv('anomaly_data.csv', index=False)
print('Data exported to CSV')


# Method-1 "Using data visualization" for the first 1000 rows
plt.plot(df['Time'][:1000], df['Metric'][:1000])
plt.title('Simulated Data with Anomalies (45% Anomaly) - First 1000 Rows')
plt.xlabel('Time')
plt.ylabel('Metric Value')
plt.scatter(df[df['Anomaly'] == 1]['Time'][:1000], df[df['Anomaly'] == 1]['Metric'][:1000], color='red', label='Anomalies')
plt.legend()
plt.show()





