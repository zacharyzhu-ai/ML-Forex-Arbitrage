# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:51:35 2024

@author: 1055842
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


fred_data = pd.read_csv('FRED euro to dollar (2014-2024).csv')  # Replace with new file name
fred_data.rename(columns={'DATE': 'Date', 'DEXUSEU': 'Exchange_Rate'}, inplace=True)  #Replace the second column with new name
fred_data['Date'] = pd.to_datetime(fred_data['Date'])

# Replace periods with NaN and convert to numeric
fred_data['Exchange_Rate'] = pd.to_numeric(fred_data['Exchange_Rate'], errors='coerce')

# Handle missing values by forward filling
fred_data['Exchange_Rate'].fillna(method='ffill', inplace=True)  # Forward fill
fred_data.set_index('Date', inplace=True)

#Calculates moving average and volatility
fred_data['Moving_Avg'] = fred_data['Exchange_Rate'].rolling(window=30).mean()
fred_data['Volatility'] = fred_data['Exchange_Rate'].rolling(window=30).std() * 5

# Add lagged values
fred_data['Lagged_1'] = fred_data['Exchange_Rate'].shift(1)  # Exchange rate from the previous day
fred_data['Lagged_2'] = fred_data['Exchange_Rate'].shift(2)  # Exchange rate from 2 days ago

# Add rate of change
fred_data['Rate_of_Change'] = fred_data['Exchange_Rate'].pct_change()  # Percentage change

# Add the target: next day's exchange rate
fred_data['Target'] = fred_data['Exchange_Rate'].shift(-1)

# Plot the exchange rate
plt.figure(figsize=(12, 6))
plt.plot(fred_data.index, fred_data['Exchange_Rate'], label='Exchange Rate', alpha=0.7)
plt.plot(fred_data.index, fred_data['Moving_Avg'], label='30-Day Moving Average', linestyle='--')
plt.plot(fred_data.index, fred_data['Volatility'], label='30-Day Volatility',color = 'g', linestyle=':', alpha=1)
plt.title('U.S. Dollars to Euro Spot Exchange Rate (2014-2024)')
plt.xlabel('Date')
plt.ylabel('Exchange Rate / Volatility')
plt.legend()
plt.show()


fred_data.to_csv('preprocessed_fred_data.csv')