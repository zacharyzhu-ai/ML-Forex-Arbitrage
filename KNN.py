# -*- coding: utf-8 -*-
"""
KNN Model for Forex Prediction
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def train_knn(fred_data, n_neighbors=3):
    """
    Trains a K-Nearest Neighbors (KNN) regression model on forex data.
    
    Args:
        fred_data (pd.DataFrame): Preprocessed dataset with features.
        n_neighbors (int): Number of neighbors to use for KNN.
    
    Returns:
        tuple: (Trained model, scaler, X_test_scaled, y_test)
    """
    # Drop any rows with missing values
    fred_data.dropna(inplace=True)

    # Features and target
    X = fred_data[['Moving_Avg', 'Lagged_1', 'Lagged_2']]
    y = fred_data['Exchange_Rate']  

    # Initialize scaler
    scaler = StandardScaler()

    # Fit and transform the features
    X_scaled = scaler.fit_transform(X)

    # Split the dataset into training and test sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Initialize KNN regressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print the results
    print("\n🔹 KNN Model Evaluation 🔹")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Plot predicted vs actual exchange rates
    plt.figure(figsize=(10,6))
    plt.plot(y_test.values, label='Actual Exchange Rate', color='blue')
    plt.plot(y_pred, label='Predicted Exchange Rate', color='red', linestyle='--')
    plt.ylim(min(y_test.values) - 1, max(y_test.values) + 1)
    plt.title("Predicted vs Actual Exchange Rates")
    plt.xlabel("Data Points")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.show()

    # Calculate residuals
    residuals = y_test.values - y_pred

    # Plot residuals
    plt.figure(figsize=(10,6))
    plt.scatter(y_test.values, residuals, color='blue', alpha=0.5)
    plt.hlines(0, min(y_test.values), max(y_test.values), colors='red', linestyles='--')
    plt.title("Residuals Plot")
    plt.xlabel("Actual Exchange Rate")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.show()

    return knn, scaler, X_test, y_test  # Returning model and test data for ensemble
