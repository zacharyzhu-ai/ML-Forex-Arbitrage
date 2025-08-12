# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 22:12:28 2025

@author: 1055842
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from linear_regression import train_linear_regression  
from KNN import train_knn
from XGBoost import train_xgboost

def forecast_future_rates(model, fred_data, days=14):
    future_rates = []
    current_df = fred_data.copy()

    for _ in range(days):
        last_row = current_df.iloc[-1]

        new_row = {}
        new_row["Lagged_1"] = last_row["Exchange_Rate"]
        new_row["Lagged_2"] = last_row["Lagged_1"]

        recent_rates = current_df["Exchange_Rate"].tail(3)
        new_row["Moving_Avg"] = recent_rates.mean()
        new_row["Volatility"] = recent_rates.std(ddof=0)
        new_row["Rate_of_Change"] = (
            (last_row["Exchange_Rate"] - last_row["Lagged_1"]) / last_row["Lagged_1"]
            if last_row["Lagged_1"] != 0 else 0
        )

        next_X = pd.DataFrame([new_row])
        next_X = next_X[X.columns]  # Match training feature order
        next_prediction = model.predict(next_X)[0]
        future_rates.append(next_prediction)

        new_row["Exchange_Rate"] = next_prediction
        current_df = pd.concat([current_df, pd.DataFrame([new_row])], ignore_index=True)

    return future_rates

def detect_arbitrage_opportunities(future_rates, threshold=0.005):
    ops = []
    for i in range(1, len(future_rates) - 1):
        prev = future_rates[i - 1]
        curr = future_rates[i]
        next_ = future_rates[i + 1]

        # Buy low opportunity
        if curr < prev and curr < next_:
            recovery = next_ - curr
            if recovery / curr > threshold:
                ops.append((f"T+{i+1}", "BUY", curr, f"Expected rise to {next_:.4f}"))

        # Sell high opportunity
        if curr > prev and curr > next_:
            drop = curr - next_
            if drop / curr > threshold:
                ops.append((f"T+{i+1}", "SELL", curr, f"Expected drop to {next_:.4f}"))

    return ops

if __name__ == "__main__":
    # Load preprocessed dataset
    fred_data = pd.read_csv("preprocessed_fred_data.csv", index_col="Date", parse_dates=True)

    # Initialize a dictionary to store models
    models = {}

    # Train Linear Regression Model
    lin_reg, X_test_scaled, y_test = train_linear_regression(fred_data)
    models["lin_reg"] = lin_reg

    # Train KNN Model
    knn, _, _, _ = train_knn(fred_data)
    models["knn"] = knn

    # Train XGBoost Model
    xgb_model, _, _ = train_xgboost(fred_data)
    models["xgb"] = xgb_model

    # Train Decision Tree Regressor
    dt_model = DecisionTreeRegressor(max_depth=5)
    X = fred_data.drop(columns=["Target", "Exchange_Rate", "Date"], errors='ignore')
    y = fred_data["Exchange_Rate"]
    dt_model.fit(X, y)
    models["decision_tree"] = dt_model

    # Create and train ensemble
    ensemble = VotingRegressor(estimators=[
        ("knn", models["knn"]),
        ("lin_reg", models["lin_reg"]),
        ("xgb", models["xgb"]),
        ("decision_tree", models["decision_tree"])
    ])
    ensemble.fit(X, y)

    # Evaluate Ensemble
    print("\n🔹 Ensemble Model Evaluation 🔹")
    print("Ensemble Score:", ensemble.score(X_test_scaled, y_test))

    # Forecast future exchange rates
    future_rates = forecast_future_rates(ensemble, fred_data, days=14)

    print("\n🔮 Future Exchange Rate Predictions:")
    for i, rate in enumerate(future_rates, 1):
        print(f"T+{i}: {rate:.4f}")

    # Arbitrage detection
    arbitrage_ops = detect_arbitrage_opportunities(future_rates)
    if arbitrage_ops:
        print("\n💰 Arbitrage Opportunities Detected:")
        for op in arbitrage_ops:
            print(f"{op[0]} → {op[1]} at {op[2]:.4f} ({op[3]})")
    else:
        print("\n💸 No arbitrage opportunities detected.")
