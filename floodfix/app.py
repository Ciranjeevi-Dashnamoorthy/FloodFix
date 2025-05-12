from flask import Flask, render_template, jsonify
import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

app = Flask(__name__)

# --- CONFIGURATION ---
API_KEY = "9cf4a7cd158c4248a2c85238251103"
CITY = "Chennai"
FORECAST_DAYS = 7
URL = f"https://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={CITY}&days={FORECAST_DAYS}"

# Check if models exist, otherwise train and save them
MODEL_FILE_RAIN = "rain_model.pkl"
MODEL_FILE_FLOOD = "flood_model.pkl"

if not os.path.exists(MODEL_FILE_RAIN) or not os.path.exists(MODEL_FILE_FLOOD):
    # --- Create Dummy Training Data ---
    np.random.seed(42)
    data = pd.DataFrame({
        "temperature": np.random.uniform(25, 35, 300),
        "humidity": np.random.uniform(60, 90, 300),
        "pressure": np.random.uniform(990, 1020, 300),
        "wind_speed": np.random.uniform(1, 10, 300),
        "rain": np.random.exponential(scale=10, size=300).clip(0, 60),
        "flood_risk": np.random.choice([0, 1], 300, p=[0.85, 0.15])
    })

    # --- Train Models ---
    X = data.drop(columns=["rain", "flood_risk"])
    y_rain = data["rain"]
    y_flood = data["flood_risk"]

    rain_model = RandomForestRegressor()
    rain_model.fit(X, y_rain)

    flood_model = LogisticRegression()
    flood_model.fit(X, y_flood)

    # Save models
    joblib.dump(rain_model, MODEL_FILE_RAIN)
    joblib.dump(flood_model, MODEL_FILE_FLOOD)
else:
    # Load existing models
    rain_model = joblib.load(MODEL_FILE_RAIN)
    flood_model = joblib.load(MODEL_FILE_FLOOD)

# --- Fetch Weather Forecast ---
def fetch_weather_forecast():
    response = requests.get(URL)
    if response.status_code == 200:
        forecast_data = response.json()
        forecast_days = forecast_data["forecast"]["forecastday"]
        weather_list = []
        for day in forecast_days:
            daily = day["day"]
            hourly_data = []
            for hour in day["hour"]:
                hourly_data.append({
                    "time": hour["time"],
                    "temp": hour["temp_c"],
                    "humidity": hour["humidity"],
                    "pressure": hour["pressure_mb"],
                    "wind_speed": hour["wind_kph"],
                    "precip": hour["precip_mm"]
                })
            weather_features = {
                "date": day["date"],
                "temperature": daily["avgtemp_c"],
                "humidity": daily["avghumidity"],
                "pressure": daily.get("pressure_mb", 1010),
                "wind_speed": daily["maxwind_kph"],
                "rain": daily["totalprecip_mm"],
                "hourly": hourly_data,
                "lat": forecast_data["location"]["lat"],
                "lon": forecast_data["location"]["lon"]
            }
            weather_list.append(weather_features)
        return weather_list
    else:
        print("Failed to fetch data:", response.status_code)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_forecast')
def get_forecast():
    forecast_data = fetch_weather_forecast()
    if forecast_data is None:
        return jsonify({"error": "Failed to fetch forecast data"}), 500
    
    # Prepare data for prediction
    prediction_data = []
    for day in forecast_data:
        features = {
            "temperature": day["temperature"],
            "humidity": day["humidity"],
            "pressure": day["pressure"],
            "wind_speed": day["wind_speed"]
        }
        df = pd.DataFrame([features])
        
        # Predict
        predicted_rain = rain_model.predict(df)[0]
        flood_risk = flood_model.predict(df)[0]
        flood_prob = flood_model.predict_proba(df)[0][1] * 100  # Probability of flood
        
        prediction_data.append({
            "date": day["date"],
            "temperature": day["temperature"],

            "humidity": day["humidity"],
            "pressure": day["pressure"],
            "wind_speed": day["wind_speed"],
            "actual_rain": day["rain"],
            "predicted_rain": float(predicted_rain),
            "flood_risk": int(flood_risk),
            "flood_probability": float(flood_prob),
            "lat": day["lat"],
            "lon": day["lon"],
            "hourly": day["hourly"]
        })
    
    return jsonify(prediction_data)

if __name__ == '__main__':
    app.run(debug=True)