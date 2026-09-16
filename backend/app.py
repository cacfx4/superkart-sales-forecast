# ==========================================================
# SuperKart Flask Backend API
# ==========================================================
# This API loads the serialized Random Forest pipeline and
# exposes endpoints for:
#
# 1. Health Check
# 2. Single Prediction
# 3. Batch Prediction
#
# Because the entire pipeline was saved, all preprocessing
# (imputation + one-hot encoding) is automatically applied
# before predictions are generated.
# ==========================================================


# ==========================================================
# Import Required Libraries
# ==========================================================
# Flask      -> Create REST API endpoints
# request    -> Receive incoming API requests
# jsonify    -> Return responses in JSON format
# pandas     -> Convert incoming data into DataFrames
# joblib     -> Load the serialized model

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os


# ==========================================================
# Create Flask Application
# ==========================================================
# Initializes the Flask API.

superkart_api = Flask("SuperKart")


# ==========================================================
# Load Trained Model
# ==========================================================
# Load the serialized Random Forest pipeline.
#
# The saved object includes:
# - Missing value handling
# - One-Hot Encoding
# - Random Forest Regressor
#
# Loading once at startup is more efficient than loading
# the model for every request.

#model = joblib.load("/content/drive/MyDrive/Data/SuperKart/deployment_files/superKart_model_v1_0.joblib")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "superKart_model_v1_0.joblib"
)

model = joblib.load(MODEL_PATH)


# ==========================================================
# Define Required Features
# ==========================================================
# These column names MUST exactly match the columns used
# to train the model.
#
# Verify against:
# X_train.columns.tolist()

REQUIRED_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Store_Age",
    "Product_ID_Code"
]


# ==========================================================
# Home Endpoint
# ==========================================================
# Simple endpoint used to verify the API is running.

@superkart_api.get("/")
def home():

    return {
        "message": "Welcome to the SuperKart Sales Forecast API"
    }


# ==========================================================
# Health Check Endpoint
# ==========================================================
# Used by Docker, GitHub Codespaces, load balancers,
# monitoring systems, and developers to confirm that
# the API is healthy.

@superkart_api.get("/health")
def health():

    return {
        "status": "healthy"
    }


# ==========================================================
# Single Prediction Endpoint
# ==========================================================
# Endpoint:
# POST /v1/predict
#
# Accepts a JSON payload describing a single product-store
# combination and returns the predicted sales revenue.
#
# Example Input:
#
# {
#   "Product_Weight": 12.5,
#   "Product_Sugar_Content": "Regular",
#   "Product_Allocated_Area": 0.12,
#   "Product_Type": "Dairy",
#   "Product_MRP": 249.90,
#   "Store_Id": "OUT001",
#   "Store_Size": "High",
#   "Store_Location_City_Type": "Tier 1",
#   "Store_Type": "Supermarket Type1",
#   "Store_Age": 27,
#   "Product_ID_Code": "FD"
# }

@superkart_api.post("/v1/predict")
def predict_sales():

    try:

        # --------------------------------------------------
        # Read incoming JSON request
        # --------------------------------------------------
        data = request.get_json()

        # --------------------------------------------------
        # Validate Required Features
        # --------------------------------------------------
        # Check for missing features before attempting
        # prediction.

        missing_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in data
        ]

        if missing_features:

            return jsonify(
                {
                    "error": "Missing required features",
                    "missing_features": missing_features
                }
            ), 400

        # --------------------------------------------------
        # Hybrid Approach
        # --------------------------------------------------
        # Create the model input using ONLY the features
        # expected by the model.
        #
        # Benefits:
        #
        # 1. Ignores unexpected fields
        # 2. Ensures correct feature order
        # 3. Prevents accidental schema changes
        # 4. Validates all required inputs

        input_df = pd.DataFrame(
            [
                {
                    feature: data[feature]
                    for feature in REQUIRED_FEATURES
                }
            ]
        )

        # --------------------------------------------------
        # Generate Prediction
        # --------------------------------------------------

        prediction = model.predict(input_df)[0]

        # --------------------------------------------------
        # Return Prediction
        # --------------------------------------------------

        return jsonify(
            {
                "predicted_sales": round(float(prediction), 2)
            }
        )

    except Exception as e:

        # --------------------------------------------------
        # Return Error Message
        # --------------------------------------------------

        return jsonify(
            {
                "error": str(e)
            }
        ), 400


# ==========================================================
# Batch Prediction Endpoint
# ==========================================================
# Endpoint:
# POST /v1/predictbatch
#
# Accepts a CSV file containing multiple records and
# returns a prediction for every row.
#
# Expected form-data:
#
# key = file
# value = uploaded CSV

@superkart_api.post("/v1/predictbatch")
def predict_sales_batch():

    try:

        # --------------------------------------------------
        # Retrieve Uploaded CSV File
        # --------------------------------------------------

        file = request.files["file"]

        # --------------------------------------------------
        # Convert CSV to DataFrame
        # --------------------------------------------------

        input_df = pd.read_csv(file)

        # --------------------------------------------------
        # Validate Required Columns
        # --------------------------------------------------

        missing_columns = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in input_df.columns
        ]

        if missing_columns:

            return jsonify(
                {
                    "error": "Missing required columns",
                    "missing_columns": missing_columns
                }
            ), 400

        # --------------------------------------------------
        # Keep Only Expected Columns
        # --------------------------------------------------

        input_df = input_df[REQUIRED_FEATURES]

        # --------------------------------------------------
        # Generate Predictions
        # --------------------------------------------------

        predictions = model.predict(input_df)

        # --------------------------------------------------
        # Format Results
        # --------------------------------------------------

        results = [
            round(float(pred), 2)
            for pred in predictions
        ]

        # --------------------------------------------------
        # Return Predictions
        # --------------------------------------------------

        return jsonify(
            {
                "predictions": results
            }
        )

    except Exception as e:

        # --------------------------------------------------
        # Return Error Message
        # --------------------------------------------------

        return jsonify(
            {
                "error": str(e)
            }
        ), 400


# ==========================================================
# Run Flask Application
# ==========================================================
# host="0.0.0.0"
# Allows external access from Docker containers,
# GitHub Codespaces, and Streamlit frontend.
#
# port=5000
# Backend API port.
#
# debug=False
# Recommended for deployment.

if __name__ == "__main__":

    superkart_api.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
