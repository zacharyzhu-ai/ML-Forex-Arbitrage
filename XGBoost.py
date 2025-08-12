# -*- coding: utf-8 -*-
"""
XGBoost Model for Forex Prediction
"""

import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def train_xgboost(fred_data, test_size=0.2, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Trains an XGBoost regression model on forex data.

    Args:
        fred_data (pd.DataFrame): Preprocessed dataset with features.
        test_size (float): Proportion of the dataset to be used as test data.
        n_estimators (int): Number of boosting rounds.
        learning_rate (float): Step size shrinkage to prevent overfitting.
        max_depth (int): Maximum depth of a tree.

    Returns:
        tuple: (Trained model, X_test, y_test)
    """

    # Ensure data does not contain missing values
    fred_data.ffill(inplace=True)  # Forward-fill missing values

    # Define features and target variable
    features = ["Moving_Avg", "Volatility", "Lagged_1", "Rate_of_Change"]
    target = "Exchange_Rate"  # Ensure target column matches other models

    # Drop remaining NaNs before training
    final_df = fred_data.dropna(subset=features + [target])

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        final_df[features], final_df[target], test_size=test_size, random_state=42
    )

    # Initialize XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        reg_alpha=0.1,  # L1 Regularization
        reg_lambda=0.1,  # L2 Regularization
        random_state=42
    )

    # Train the model
    xgb_model.fit(X_train, y_train)

    # Predictions on train set
    y_train_pred = xgb_model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)

    # Predictions on test set
    y_pred = xgb_model.predict(X_test)

    # Evaluate model performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print evaluation metrics
    print("\n🔹 XGBoost Model Evaluation 🔹")
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R² Score: {r2:.4f}")

    # Feature Importance Plot
    xgb.plot_importance(xgb_model)
    plt.title("Feature Importance - XGBoost")
    plt.show()

    return xgb_model, X_test, y_test  # Returning model and test data
