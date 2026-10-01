from flask import Flask, render_template, request

import pandas as pd

from src.prediction import predict_sepsis


app = Flask(
    __name__,
    template_folder="template",
    static_folder="static",
)


@app.route("/")
def home():
    return render_template(
        "index.html"
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:

        patient_data = {
            "age": float(request.form["age"]),
            "gender": request.form["gender"],
            "bmi": float(request.form["bmi"]),

            "hr_mean": float(request.form["hr_mean"]),
            "sbp_mean": float(request.form["sbp_mean"]),
            "dbp_mean": float(request.form["dbp_mean"]),

            "temp_celsius_mean": float(
                request.form["temp_celsius_mean"]
            ),

            "respiratory_rate_mean": float(
                request.form["respiratory_rate_mean"]
            ),

            "spo2_mean": float(
                request.form["spo2_mean"]
            ),

            "map_mean": float(
                request.form["map_mean"]
            ),

            "pao2_fio2_ratio": float(
                request.form["pao2_fio2_ratio"]
            ),

            "bilirubin_total": float(
                request.form["bilirubin_total"]
            ),

            "ph_arterial": float(
                request.form["ph_arterial"]
            ),

            "sodium": float(
                request.form["sodium"]
            ),

            "gcs_total": float(
                request.form["gcs_total"]
            ),

            "creatinine": float(
                request.form["creatinine"]
            ),

            "chronic_kidney_disease": int(
                request.form["chronic_kidney_disease"]
            ),

            "wbc": float(
                request.form["wbc"]
            ),

            "platelet_count": float(
                request.form["platelet_count"]
            ),

            "hemoglobin": float(
                request.form["hemoglobin"]
            )
        }

        patient_df = pd.DataFrame(
            [patient_data]
        )

        result, probability = predict_sepsis(
            patient_df
        )

        probability = round(
            probability * 100,
            2
        )

        return render_template(
            "index.html",
            prediction=result,
            probability=probability
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(
        debug=True
    )