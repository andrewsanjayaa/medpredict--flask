from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ── Load model bundle ────────────────────────────────────────────────────────
def load_model():
    if os.path.exists("model.pkl"):
        bundle = joblib.load("model.pkl")

        # Kalau model.pkl berupa bundle dictionary
        if isinstance(bundle, dict):
            return bundle

        # Kalau model.pkl berupa model langsung
        return {
            "model": bundle,
            "scaler": None,
            "feature_columns": None
        }

    raise FileNotFoundError("model.pkl tidak ditemukan.")

model_bundle = load_model()
model = model_bundle["model"]
scaler = model_bundle.get("scaler", None)
feature_columns = model_bundle.get("feature_columns", None)


@app.route("/")
def home():
    return render_template("index.html", page="home")


@app.route("/assessment")
def assessment():
    return render_template("index.html", page="assessment")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ambil data dari form
        age = float(data["age"])
        gender = data["gender"]
        bp_systolic = float(data["bp_systolic"])
        bp_diastolic = float(data["bp_diastolic"])
        cholesterol = float(data["cholesterol"])
        bmi = float(data["bmi"])
        diabetes = int(data["diabetes"])
        hypertension = int(data["hypertension"])
        medication_count = int(data["medication_count"])
        length_of_stay = float(data["length_of_stay"])
        discharge_destination = data["discharge_destination"]

        # Buat dataframe mentah
        input_df = pd.DataFrame([{
            "age": age,
            "BMI": bmi,
            "cholesterol": cholesterol,
            "systolic": bp_systolic,
            "diastolic": bp_diastolic,
            "pulse_pressure": bp_systolic - bp_diastolic,
            "diabetes": diabetes,
            "hypertension": hypertension,
            "medication_count": medication_count,
            "length_of_stay": length_of_stay,
            "gender_Male": 1 if gender == "Male" else 0,
            "discharge_destination_Nursing_Facility": 1 if discharge_destination == "SNF" else 0,
            "discharge_destination_Rehab": 1 if discharge_destination == "Rehab" else 0
        }])

        # Samakan urutan kolom dengan waktu training
        if feature_columns is not None:
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0

            input_df = input_df[feature_columns]

        # Scaling kalau scaler tersedia
        if scaler is not None:
            features = scaler.transform(input_df)
        else:
            features = input_df.values

        # Prediksi
        pred = int(model.predict(features)[0])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = round(float(max(proba)) * 100, 1)
        else:
            confidence = 0

        actions_yes = [
            "Schedule follow-up within 48 hours of discharge.",
            "Initiate home-health nursing evaluation.",
            "Comprehensive medication reconciliation review.",
            "Enroll patient in remote tele-monitoring program.",
        ]

        actions_no = [
            "Standard discharge protocol applicable.",
            "Routine 30-day follow-up appointment.",
            "Patient education on warning signs.",
        ]

        return jsonify({
            "prediction": pred,
            "confidence": confidence,
            "actions": actions_yes if pred == 1 else actions_no,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)