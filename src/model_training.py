import os
from pathlib import Path

import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from data_preprocessing import preprocess_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data():
    """
    Load the cleaned/processed dataset.
    """

    data_path = DATA_DIR / "cleaned_sepsis_data.csv"

    return str(data_path)


def train_model(X_train, y_train):
    """
    Train Decision Tree classifier.
    """

    model = DecisionTreeClassifier(
        random_state=42,
        criterion='gini',
        max_depth=3,
        min_samples_split=3,
        min_samples_leaf=1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model.
    """

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nModel Evaluation")
    print("----------------")

    print(
        "Accuracy:",
        accuracy
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )


def save_model(model):
    """
    Save the trained Decision Tree model.
    """

    os.makedirs(
        MODELS_DIR,
        exist_ok=True
    )

    model_path = MODELS_DIR / "final_model.pkl"

    joblib.dump(
        model,
        model_path
    )

    print(
        f"\nFinal model saved to: {model_path}"
    )


def main():

    print("Starting model training...")

    # Load original dataset
    import pandas as pd

    data_path = DATA_DIR / "sepsis_icu_synthetic.csv"

    df = pd.read_csv(
        data_path
    )

    print(
        "Dataset loaded successfully."
    )

    # Preprocessing
    (
        X_train,
        X_test,
        y_train,
        y_test,
        gender_encoder,
        scaler,
        feature_columns
    ) = preprocess_pipeline(df)

    print(
        "\nPreprocessing pipeline completed."
    )

    # Train Decision Tree
    model = train_model(
        X_train,
        y_train
    )

    print(
        "\nDecision Tree training completed."
    )

    # Evaluate
    evaluate_model(
        model,
        X_test,
        y_test
    )

    # Save final model
    save_model(
        model
    )


if __name__ == "__main__":
    main()