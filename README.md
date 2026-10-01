# Sepsis Risk Prediction with Explainable AI

This project develops an intelligent web application to predict the risk of sepsis in ICU patients using clinical and physiological markers. The system takes patient information such as age, blood pressure, heart rate, oxygen saturation, kidney function, blood counts, and other vitals, then applies a trained machine learning model to classify whether the patient is likely to have sepsis.

The application is built using Python, Flask, and a decision tree classifier. It is designed to support early clinical risk assessment and provide a quick, interpretable decision support tool for healthcare professionals.

## Features

- Predicts sepsis risk from patient health parameters
- Uses a trained ML model for classification
- Provides probability output for the predicted class
- Includes preprocessing and feature engineering for medical data
- Offers a simple web interface for entering patient inputs
- Uses explainability-related components to support transparent decision-making

## Project Structure

- `app.py` – Flask web application and prediction endpoint
- `src/data_ingestion.py` – data loading logic
- `src/data_preprocessing.py` – preprocessing and cleaning pipeline
- `src/model_training.py` – model training and evaluation
- `src/prediction.py` – prediction logic and risk scoring
- `src/explainability.py` – explainability-related functionality
- `data/` – dataset files
- `models/` – trained model artifacts
- `template/` – HTML front-end templates
- `static/` – CSS and front-end assets
- `notebooks/` – exploratory notebooks and experiments
- `requirements.txt` – project dependencies

## Tech Stack

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- HTML/CSS

## Installation

1. Clone the repository
2. Create a virtual environment
3. Install dependencies:

```bash
pip install -r requirements.txt