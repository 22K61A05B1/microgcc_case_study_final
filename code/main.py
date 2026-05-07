import os

# =========================================
# TENSORFLOW WARNING SUPPRESSION
# =========================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import logging

# Disable Prophet logs
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("prophet").disabled = True

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet import Prophet

from xgboost import XGBRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import holidays
import joblib

# =========================================
# CREATE REQUIRED FOLDERS
# =========================================

os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/future_forecasts", exist_ok=True)

# =========================================
# LOAD DATASET
# =========================================

print("Loading dataset...")

file_path = "data/sales_data.xlsx"

df = pd.read_excel(file_path)

print("\nDataset Loaded Successfully!")
print(df.head())

# =========================================
# COLUMN NAMES
# =========================================

DATE_COLUMN = "Date"
STATE_COLUMN = "State"
SALES_COLUMN = "Total"

# =========================================
# DATA PREPROCESSING
# =========================================

print("\nPreprocessing data...")

# Convert to datetime
df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])

# Sort by date
df = df.sort_values(by=DATE_COLUMN)

# Remove duplicates
df = df.drop_duplicates()

# Fill missing sales values
df[SALES_COLUMN] = df[SALES_COLUMN].ffill()

# =========================================
# FEATURE ENGINEERING
# =========================================

print("\nCreating features...")

# Date features
df["day"] = df[DATE_COLUMN].dt.day
df["month"] = df[DATE_COLUMN].dt.month
df["dayofweek"] = df[DATE_COLUMN].dt.dayofweek
df["weekofyear"] = df[DATE_COLUMN].dt.isocalendar().week.astype(int)

# Holiday feature
india_holidays = holidays.India()

df["holiday"] = df[DATE_COLUMN].apply(
    lambda x: 1 if x in india_holidays else 0
)

# Lag Features
df["lag_1"] = df[SALES_COLUMN].shift(1)
df["lag_7"] = df[SALES_COLUMN].shift(7)
df["lag_30"] = df[SALES_COLUMN].shift(30)

# Rolling Features
df["rolling_mean"] = df[SALES_COLUMN].rolling(window=7).mean()
df["rolling_std"] = df[SALES_COLUMN].rolling(window=7).std()

# Remove null rows
df = df.dropna()

print("Feature Engineering Completed!")

# =========================================
# STORE RESULTS
# =========================================

results = []

# =========================================
# PROCESS EACH STATE
# =========================================

states = df[STATE_COLUMN].unique()

for state in states:

    print("\n===================================")
    print(f"Processing State: {state}")
    print("===================================")

    # Filter state data
    state_df = df[df[STATE_COLUMN] == state].copy()

    # Sort by date
    state_df = state_df.sort_values(by=DATE_COLUMN)

    # =========================================
    # TRAIN TEST SPLIT
    # =========================================

    split_index = int(len(state_df) * 0.8)

    train = state_df.iloc[:split_index]
    test = state_df.iloc[split_index:]

    y_train = train[SALES_COLUMN]
    y_test = test[SALES_COLUMN]

    # =========================================
    # SARIMA MODEL
    # =========================================

    print("\nTraining SARIMA...")

    try:

        sarima_train = train[
            [DATE_COLUMN, SALES_COLUMN]
        ].copy()

        sarima_train = sarima_train.set_index(
            DATE_COLUMN
        )

        sarima_train = sarima_train.asfreq("MS")

        sarima_train[SALES_COLUMN] = (
            sarima_train[SALES_COLUMN].ffill()
        )

        sarima_model = SARIMAX(
            sarima_train[SALES_COLUMN],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )

        sarima_fit = sarima_model.fit(
            disp=False
        )

        sarima_pred = sarima_fit.forecast(
            steps=len(y_test)
        )

        sarima_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                sarima_pred
            )
        )

        print("SARIMA RMSE:", sarima_rmse)

    except Exception as e:

        print("SARIMA Error:", e)

        sarima_rmse = 999999

    # =========================================
    # PROPHET MODEL
    # =========================================

    print("\nTraining Prophet...")

    try:

        prophet_train = train[
            [DATE_COLUMN, SALES_COLUMN]
        ].copy()

        prophet_train.columns = ["ds", "y"]

        prophet_model = Prophet()

        prophet_model.fit(prophet_train)

        future = prophet_model.make_future_dataframe(
            periods=len(test)
        )

        forecast = prophet_model.predict(future)

        prophet_pred = forecast["yhat"].tail(
            len(test)
        ).values

        prophet_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                prophet_pred
            )
        )

        print("Prophet RMSE:", prophet_rmse)

    except Exception as e:

        print("Prophet Error:", e)

        prophet_rmse = 999999

    # =========================================
    # XGBOOST MODEL
    # =========================================

    print("\nTraining XGBoost...")

    try:

        feature_cols = [
            "day",
            "month",
            "dayofweek",
            "weekofyear",
            "holiday",
            "lag_1",
            "lag_7",
            "lag_30",
            "rolling_mean",
            "rolling_std"
        ]

        X_train = train[feature_cols]
        X_test = test[feature_cols]

        xgb_model = XGBRegressor()

        xgb_model.fit(X_train, y_train)

        xgb_pred = xgb_model.predict(X_test)

        xgb_rmse = np.sqrt(
            mean_squared_error(
                y_test,
                xgb_pred
            )
        )

        print("XGBoost RMSE:", xgb_rmse)

    except Exception as e:

        print("XGBoost Error:", e)

        xgb_rmse = 999999

    # =========================================
    # LSTM MODEL
    # =========================================

    print("\nTraining LSTM...")

    try:

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(
            state_df[[SALES_COLUMN]]
        )

        X = []
        y = []

        sequence_length = 7

        for i in range(
            sequence_length,
            len(scaled_data)
        ):

            X.append(
                scaled_data[
                    i-sequence_length:i
                ]
            )

            y.append(scaled_data[i])

        X = np.array(X)
        y = np.array(y)

        split = int(len(X) * 0.8)

        X_train_lstm = X[:split]
        X_test_lstm = X[split:]

        y_train_lstm = y[:split]
        y_test_lstm = y[split:]

        model = Sequential()

        model.add(
            LSTM(
                50,
                activation="relu",
                input_shape=(
                    X_train_lstm.shape[1],
                    1
                )
            )
        )

        model.add(Dense(1))

        model.compile(
            optimizer="adam",
            loss="mse"
        )

        model.fit(
            X_train_lstm,
            y_train_lstm,
            epochs=10,
            verbose=0
        )

        lstm_pred = model.predict(
            X_test_lstm,
            verbose=0
        )

        lstm_pred = scaler.inverse_transform(
            lstm_pred
        )

        y_test_actual = scaler.inverse_transform(
            y_test_lstm
        )

        lstm_rmse = np.sqrt(
            mean_squared_error(
                y_test_actual,
                lstm_pred
            )
        )

        print("LSTM RMSE:", lstm_rmse)

    except Exception as e:

        print("LSTM Error:", e)

        lstm_rmse = 999999

    # =========================================
    # MODEL COMPARISON
    # =========================================

    model_scores = {
        "SARIMA": sarima_rmse,
        "Prophet": prophet_rmse,
        "XGBoost": xgb_rmse,
        "LSTM": lstm_rmse
    }

    best_model = min(
        model_scores,
        key=model_scores.get
    )

    print("\nBest Model:", best_model)

    # =========================================
    # FUTURE 8-WEEK FORECAST
    # =========================================

    future_predictions = []

    try:

        if best_model == "SARIMA":

            future_pred = sarima_fit.forecast(steps=8)

            future_predictions = future_pred.values

        elif best_model == "Prophet":

            future_future = prophet_model.make_future_dataframe(
                periods=8,
                freq="W"
            )

            future_forecast = prophet_model.predict(
                future_future
            )

            future_predictions = future_forecast[
                "yhat"
            ].tail(8).values

        elif best_model == "XGBoost":

            latest_row = state_df.iloc[-1:].copy()

            for i in range(8):

                future_value = xgb_model.predict(
                    latest_row[feature_cols]
                )[0]

                future_predictions.append(
                    future_value
                )

        elif best_model == "LSTM":

            last_sequence = scaled_data[
                -sequence_length:
            ]

            current_sequence = last_sequence.copy()

            for i in range(8):

                pred = model.predict(
                    current_sequence.reshape(
                        1,
                        sequence_length,
                        1
                    ),
                    verbose=0
                )

                predicted_value = scaler.inverse_transform(
                    pred
                )[0][0]

                future_predictions.append(
                    predicted_value
                )

                current_sequence = np.append(
                    current_sequence[1:],
                    pred
                )

        future_df = pd.DataFrame({
            "Week": range(1, 9),
            "Predicted_Sales": future_predictions
        })

        future_df.to_csv(
            f"outputs/future_forecasts/{state}_future.csv",
            index=False
        )

    except Exception as e:

        print("Future Forecast Error:", e)

    # =========================================
    # VISUALIZATION
    # =========================================

    try:

        plt.figure(figsize=(10, 5))

        plt.plot(
            y_test.values,
            label="Actual"
        )

        if best_model == "SARIMA":

            plt.plot(
                sarima_pred.values,
                label="Predicted"
            )

        elif best_model == "Prophet":

            plt.plot(
                prophet_pred,
                label="Predicted"
            )

        elif best_model == "XGBoost":

            plt.plot(
                xgb_pred,
                label="Predicted"
            )

        elif best_model == "LSTM":

            plt.plot(
                lstm_pred.flatten(),
                label="Predicted"
            )

        plt.title(
            f"{state} - {best_model} Forecast"
        )

        plt.xlabel("Time")

        plt.ylabel("Sales")

        plt.legend()

        plt.savefig(
            f"outputs/plots/{state}_forecast.png"
        )

        plt.close()

    except Exception as e:

        print("Plot Error:", e)

    # =========================================
    # SAVE BEST MODEL
    # =========================================

    try:

        if best_model == "XGBoost":

            joblib.dump(
                xgb_model,
                f"outputs/models/{state}_xgboost.pkl"
            )

        elif best_model == "LSTM":

            model.save(
                f"outputs/models/{state}_lstm.keras"
            )

    except Exception as e:

        print("Model Save Error:", e)

    # =========================================
    # STORE RESULTS
    # =========================================

    results.append({
        "State": state,
        "Best_Model": best_model,
        "SARIMA_RMSE": sarima_rmse,
        "Prophet_RMSE": prophet_rmse,
        "XGBoost_RMSE": xgb_rmse,
        "LSTM_RMSE": lstm_rmse
    })

# =========================================
# SAVE FINAL RESULTS
# =========================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/model_results.csv",
    index=False
)

# =========================================
# RMSE COMPARISON CHART
# =========================================

results_df.set_index("State")[[
    "SARIMA_RMSE",
    "Prophet_RMSE",
    "XGBoost_RMSE",
    "LSTM_RMSE"
]].plot(
    kind="bar",
    figsize=(20, 8)
)

plt.title("RMSE Comparison Across Models")

plt.ylabel("RMSE")

plt.tight_layout()

plt.savefig(
    "outputs/rmse_comparison.png"
)

plt.close()

# =========================================
# FINAL OUTPUT
# =========================================

print("\n===================================")
print("MODEL COMPARISON COMPLETED!")
print("===================================")

print("\nResults Saved To:")
print("outputs/model_results.csv")

print("\nPlots Saved To:")
print("outputs/plots/")

print("\nModels Saved To:")
print("outputs/models/")

print("\nFuture Forecasts Saved To:")
print("outputs/future_forecasts/")

print("\nRMSE Chart Saved To:")
print("outputs/rmse_comparison.png")

print("\nFinal Results:")
print(results_df)