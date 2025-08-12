# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 23:56:43 2025

@author: 1055842
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_linear_regression(fred_data):
    """
    Trains a Linear Regression model on forex data.

    Args:
        fred_data (pd.DataFrame): Preprocessed dataset with features.

    Returns:
        tuple: (Trained model, X_test, y_test)
    """
    # Define features and target
    X = fred_data[['Moving_Avg', 'Volatility', 'Lagged_1', 'Lagged_2', 'Rate_of_Change']]
    y = fred_data['Exchange_Rate']  # Target variable should be continuous

    # Drop missing values
    X = X.dropna()
    y = y.loc[X.index]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression Model
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # Evaluate model
    y_pred = lin_reg.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n🔹 Linear Regression Model Evaluation 🔹")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return lin_reg, X_test, y_test  # Return trained model
