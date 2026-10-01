from pathlib import Path

import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_model():
    """
    Load the trained Decision Tree model.
    """

    model_path = MODELS_DIR / "final_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {model_path}. "
            "Train the model first with: python src/model_training.py"
        )

    model = joblib.load(model_path)

    return model


def create_explainer(model):
    """
    Create SHAP TreeExplainer.
    """

    explainer = shap.TreeExplainer(
        model
    )

    return explainer


def global_explanation(X_test):
    """
    Show global feature importance.
    """

    model = load_model()

    explainer = create_explainer(
        model
    )

    shap_values = explainer.shap_values(
        X_test
    )
    if isinstance(shap_values, list):
        sepsis_shap_values = shap_values[1]
    else:
        sepsis_shap_values = shap_values

    shap.summary_plot(
        sepsis_shap_values,
        X_test,
        show=True
    )


def local_explanation(X_test, patient_index=0):

    model = load_model()

    explainer = create_explainer(
        model
    )

    patient = X_test.iloc[
        [patient_index]
    ]

    shap_values = explainer.shap_values(
        patient
    )

    if isinstance(shap_values, list):
        sepsis_shap_values = shap_values[1]
        base_value = explainer.expected_value[1]
    else:
        sepsis_shap_values = shap_values
        base_value = explainer.expected_value

    shap.force_plot(
        base_value,
        sepsis_shap_values[0],
        patient.iloc[0],
        matplotlib=True
    )

    plt.show()


if __name__ == "__main__":

    # Load test data
    X_test = pd.read_csv(
        "data/X_test.csv"
    )

    print(
        "Generating global SHAP explanation..."
    )

    global_explanation(
        X_test
    )

    print(
        "Generating local SHAP explanation..."
    )

    local_explanation(
        X_test,
        patient_index=0
    )