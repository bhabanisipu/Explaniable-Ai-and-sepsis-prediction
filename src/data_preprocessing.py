import os
from pathlib import Path

import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


def clean_data(df):
    """
    Perform basic data cleaning on the raw dataset.
    """

    cleaned_df = df.copy()

    # Fix incorrect gender value
    cleaned_df["gender"] = cleaned_df["gender"].replace({
        "Mael": "M"
    })

    # Invalid SpO2 maximum values
    cleaned_df.loc[
        cleaned_df["spo2_max"] > 100,
        "spo2_max"
    ] = np.nan

    # Invalid temperature mean values
    invalid_temp_mean = (
        (cleaned_df["temp_celsius_mean"] < cleaned_df["temp_celsius_min"]) |
        (cleaned_df["temp_celsius_mean"] > cleaned_df["temp_celsius_max"])
    )

    cleaned_df.loc[
        invalid_temp_mean,
        "temp_celsius_mean"
    ] = np.nan

    # Invalid heart-rate mean values
    invalid_hr_mean = (
        (cleaned_df["hr_mean"] < cleaned_df["hr_min"]) |
        (cleaned_df["hr_mean"] > cleaned_df["hr_max"])
    )

    cleaned_df.loc[
        invalid_hr_mean,
        "hr_mean"
    ] = np.nan

    # Invalid systolic BP mean values
    invalid_sbp_mean = (
        (cleaned_df["sbp_mean"] < cleaned_df["sbp_min"]) |
        (cleaned_df["sbp_mean"] > cleaned_df["sbp_max"])
    )

    cleaned_df.loc[
        invalid_sbp_mean,
        "sbp_mean"
    ] = np.nan

    # Invalid diastolic BP mean values
    invalid_dbp_mean = (
        (cleaned_df["dbp_mean"] < cleaned_df["dbp_min"]) |
        (cleaned_df["dbp_mean"] > cleaned_df["dbp_max"])
    )

    cleaned_df.loc[
        invalid_dbp_mean,
        "dbp_mean"
    ] = np.nan

    return cleaned_df



def create_features(df):
    """
    Create engineered features for sepsis prediction.
    """

    feature_df = df.copy()

    # 1. Shock Index
    feature_df["shock_index"] = (
        feature_df["hr_mean"] /
        feature_df["sbp_mean"]
    )

    # 2. Pulse Pressure
    feature_df["pulse_pressure"] = (
        feature_df["sbp_mean"] -
        feature_df["dbp_mean"]
    )

    # 3. Oxygen-Respiratory Risk
    feature_df["oxygen_respiratory_risk"] = (
        feature_df["respiratory_rate_mean"] /
        feature_df["spo2_mean"]
    )

    # 4. Kidney Risk
    feature_df["kidney_risk"] = (
        (feature_df["creatinine"] > 1.3).astype(int) |
        (feature_df["chronic_kidney_disease"] == 1).astype(int)
    ).astype(int)

    # 5. Blood Abnormality Score
    feature_df["blood_abnormality_score"] = (
        (feature_df["wbc"] > 12000).astype(int) +
        (feature_df["wbc"] < 4000).astype(int) +
        (feature_df["platelet_count"] < 150000).astype(int) +
        (feature_df["hemoglobin"] < 12).astype(int)
    )

    return feature_df

def select_features(df):
    """
    Select the final features used by the model.
    """

    selected_features = [
        "age",
        "gender",
        "bmi",
        "shock_index",
        "pulse_pressure",
        "temp_celsius_mean",
        "oxygen_respiratory_risk",
        "map_mean",
        "pao2_fio2_ratio",
        "bilirubin_total",
        "ph_arterial",
        "sodium",
        "gcs_total",
        "kidney_risk",
        "blood_abnormality_score",
        "sepsis_label"
    ]

    selected_df = df[selected_features].copy()

    return selected_df


def split_data(df):
    """
    Split the dataset into training and testing data.
    """

    X = df.drop(columns=["sepsis_label"])
    y = df["sepsis_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test



def preprocess_data(X_train, X_test):
    """
    Preprocess training and testing data without data leakage.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    # Categorical column
    categorical_columns = [
        "gender"
    ]

    # Numerical columns
    numerical_columns = [
        column
        for column in X_train.columns
        if column not in categorical_columns
    ]


    for column in numerical_columns:

        median_value = X_train[column].median()

        X_train[column] = X_train[column].fillna(
            median_value
        )

        X_test[column] = X_test[column].fillna(
            median_value
        )


    for column in categorical_columns:

        mode_value = X_train[column].mode()[0]

        X_train[column] = X_train[column].fillna(
            mode_value
        )

        X_test[column] = X_test[column].fillna(
            mode_value
        )


    continuous_columns = [
        "age",
        "bmi",
        "shock_index",
        "pulse_pressure",
        "temp_celsius_mean",
        "oxygen_respiratory_risk",
        "map_mean",
        "pao2_fio2_ratio",
        "bilirubin_total",
        "ph_arterial",
        "sodium"
    ]

    for column in continuous_columns:

        Q1 = X_train[column].quantile(0.25)

        Q3 = X_train[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR

        upper_bound = Q3 + 1.5 * IQR

        # Apply training bounds to training data
        X_train[column] = X_train[column].clip(
            lower=lower_bound,
            upper=upper_bound
        )

        # Apply SAME training bounds to test data
        X_test[column] = X_test[column].clip(
            lower=lower_bound,
            upper=upper_bound
        )


    gender_encoder = LabelEncoder()

    X_train["gender"] = gender_encoder.fit_transform(
        X_train["gender"]
    )

    X_test["gender"] = gender_encoder.transform(
        X_test["gender"]
    )


    scaler = StandardScaler()

    X_train[numerical_columns] = scaler.fit_transform(
        X_train[numerical_columns]
    )

    X_test[numerical_columns] = scaler.transform(
        X_test[numerical_columns]
    )

    return (
        X_train,
        X_test,
        gender_encoder,
        scaler
    )


def apply_smote(X_train, y_train):
    """
    Apply SMOTE only to the training data.
    """

    smote = SMOTE(
        random_state=42
    )

    X_train_smote, y_train_smote = smote.fit_resample(
        X_train,
        y_train
    )

    return (
        X_train_smote,
        y_train_smote
    )



def preprocess_pipeline(df):
    """
    Complete preprocessing pipeline.

    Steps:
    1. Clean data
    2. Feature engineering
    3. Feature selection
    4. Save cleaned dataset
    5. Train-test split
    6. Missing value handling
    7. IQR outlier capping
    8. Gender encoding
    9. Standard scaling
    10. Save preprocessing objects
    11. Apply SMOTE
    """


    df = clean_data(df)

    print("Data cleaning completed.")

    df = create_features(df)

    print("Feature engineering completed.")


    df = select_features(df)

    print("Feature selection completed.")


    os.makedirs(
        "data",
        exist_ok=True
    )

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    models_dir = project_root / "models"

    cleaned_data_path = data_dir / "cleaned_sepsis_data.csv"

    df.to_csv(
        cleaned_data_path,
        index=False
    )

    print(
        f"Cleaned dataset saved to: {cleaned_data_path}"
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(df)

    print("Train-test split completed.")

    print(
        "Training data shape:",
        X_train.shape
    )

    print(
        "Testing data shape:",
        X_test.shape
    )


    (
        X_train,
        X_test,
        gender_encoder,
        scaler
    ) = preprocess_data(
        X_train,
        X_test
    )

    print("Preprocessing completed.")


    os.makedirs(
        models_dir,
        exist_ok=True
    )

    # Save gender encoder
    joblib.dump(
        gender_encoder,
        models_dir / "gender_encoder.pkl"
    )

    # Save scaler
    joblib.dump(
        scaler,
        models_dir / "scaler.pkl"
    )

    # Save feature order
    feature_columns = X_train.columns.tolist()

    joblib.dump(
        feature_columns,
        models_dir / "feature_columns.pkl"
    )

    print(
        "Gender encoder saved."
    )

    print(
        "Scaler saved."
    )

    print(
        "Feature columns saved."
    )

    (
        X_train_smote,
        y_train_smote
    ) = apply_smote(
        X_train,
        y_train
    )

    print("SMOTE completed.")

    print(
        "Training data after SMOTE:",
        X_train_smote.shape
    )

    print(
        "Class distribution after SMOTE:"
    )

    print(
        y_train_smote.value_counts()
    )


    return (
        X_train_smote,
        X_test,
        y_train_smote,
        y_test,
        gender_encoder,
        scaler,
        feature_columns
    )