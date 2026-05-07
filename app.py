from flask import Flask, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({
        "message": "Retail Forecasting API Running"
    })

@app.route("/predict/<state>")
def predict(state):
    file_path = f"outputs/future_forecasts/{state}_future.csv"

    if not os.path.exists(file_path):
        return jsonify({
            "error": "State forecast not found"
        })

    df = pd.read_csv(file_path)
    predictions = df.to_dict(orient="records")

    return jsonify({
        "state": state,
        "forecast": predictions
    })

if __name__ == "__main__":
    app.run(debug=True)