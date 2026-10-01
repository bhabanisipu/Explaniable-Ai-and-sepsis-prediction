from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def artifact_path(filename):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {path}. "
            "Train the model first with: python src/model_training.py"
        )
    return path


def load_artifacts():
    model = joblib.load(artifact_path("final_model.pkl"))
    gender_encoder = joblib.load(artifact_path("gender_encoder.pkl"))
    scaler = joblib.load(artifact_path("scaler.pkl"))
    feature_columns = joblib.load(artifact_path("feature_columns.pkl"))

    return model, gender_encoder, scaler, feature_columns


def create_features(patient_data):
    patient_data = patient_data.copy()

    patient_data["shock_index"] = (
        patient_data["hr_mean"] /
        patient_data["sbp_mean"]
    )

    patient_data["pulse_pressure"] = (
        patient_data["sbp_mean"] -
        patient_data["dbp_mean"]
    )

    patient_data["oxygen_respiratory_risk"] = (
        patient_data["respiratory_rate_mean"] /
        patient_data["spo2_mean"]
    )

    patient_data["kidney_risk"] = (
        (patient_data["creatinine"] > 1.3).astype(int) |
        (patient_data["chronic_kidney_disease"] == 1).astype(int)
    ).astype(int)

    patient_data["blood_abnormality_score"] = (
        (patient_data["wbc"] > 12000).astype(int) +
        (patient_data["wbc"] < 4000).astype(int) +
        (patient_data["platelet_count"] < 150000).astype(int) +
        (patient_data["hemoglobin"] < 12).astype(int)
    )

    return patient_data


def preprocess_input(patient_data):
    (
        model,
        gender_encoder,
        scaler,
        feature_columns
    ) = load_artifacts()

    patient_data = create_features(patient_data)

    patient_data = patient_data[feature_columns]

    categorical_columns = ["gender"]

    numerical_columns = [
        column for column in patient_data.columns
        if column not in categorical_columns
    ]

    # Fill missing numerical values
    for column in numerical_columns:
        patient_data[column] = patient_data[column].fillna(0)

    # Fill missing gender
    patient_data["gender"] = patient_data["gender"].fillna("M")

    # Encode gender
    patient_data["gender"] = gender_encoder.transform(
        patient_data["gender"]
    )

    # Scale numerical features
    patient_data[numerical_columns] = scaler.transform(
        patient_data[numerical_columns]
    )

    return patient_data, model


def predict_sepsis(patient_data):

    processed_data, model = preprocess_input(
        patient_data
    )

    prediction = model.predict(
        processed_data
    )

    probability = model.predict_proba(
        processed_data
    )

    if prediction[0] == 1:
        result = "Sepsis"
    else:
        result = "No Sepsis"

    sepsis_probability = probability[0][1]

    return result, sepsis_probability


if __name__ == "__main__":

    patient = pd.DataFrame([{
        "age": 65,
        "gender": "M",
        "bmi": 27.5,
        "hr_mean": 105,
        "sbp_mean": 90,
        "dbp_mean": 60,
        "temp_celsius_mean": 38.5,
        "respiratory_rate_mean": 25,
        "spo2_mean": 92,
        "map_mean": 70,
        "pao2_fio2_ratio": 250,
        "bilirubin_total": 1.5,
        "ph_arterial": 7.35,
        "sodium": 138,
        "gcs_total": 13,
        "creatinine": 1.8,
        "chronic_kidney_disease": 0,
        "wbc": 15000,
        "platelet_count": 130000,
        "hemoglobin": 11
    }])

    result, probability = predict_sepsis(
        patient
    )

    print("Prediction:", result)
    print(
        "Sepsis Probability:",
        round(probability * 100, 2),
        "%"
    )