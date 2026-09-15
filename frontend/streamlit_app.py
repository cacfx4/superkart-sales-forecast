# ==========================================================
# Import Required Libraries
# ==========================================================

import streamlit as st
import pandas as pd
import requests


# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="SuperKart Sales Forecast",
    page_icon="🛒",
    layout="wide"
)


# ==========================================================
# Backend Configuration
# ==========================================================
# Docker container name for Flask backend

BACKEND_URL = "http://superkart-backend:5000"


# ==========================================================
# Application Title
# ==========================================================

st.title("🛒 SuperKart Sales Forecasting System")

st.markdown("""
Forecast product sales revenue using the trained Random Forest model.

Choose one of the available options:

- **Single Prediction**
- **Batch Prediction**
""")


# ==========================================================
# Tabs
# ==========================================================

tab1, tab2 = st.tabs(
    [
        "Single Prediction",
        "Batch Prediction"
    ]
)


# ==========================================================
# SINGLE PREDICTION
# ==========================================================

with tab1:

    st.header("Predict Sales for One Product")

    col1, col2 = st.columns(2)

    with col1:

        product_weight = st.number_input(
            "Product Weight",
            min_value=0.0,
            value=10.0
        )

        product_sugar_content = st.selectbox(
            "Product Sugar Content",
            [
                "Low Sugar",
                "Regular",
                "No Sugar"
            ]
        )

        product_allocated_area = st.number_input(
            "Product Allocated Area",
            min_value=0.0,
            value=0.10
        )

        product_type = st.selectbox(
            "Product Type",
            [
                "Dairy",
                "Snack Foods",
                "Canned",
                "Frozen Foods",
                "Breads",
                "Breakfast",
                "Soft Drinks",
                "Hard Drinks",
                "Health and Hygiene",
                "Household",
                "Meat",
                "Fruits and Vegetables",
                "Seafood",
                "Baking Goods",
                "Starchy Foods",
                "Others"
            ]
        )

        product_mrp = st.number_input(
            "Product MRP",
            min_value=0.0,
            value=150.0
        )

    with col2:

        store_id = st.text_input(
            "Store ID",
            value="OUT001"
        )

        store_size = st.selectbox(
            "Store Size",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

        city_tier = st.selectbox(
            "Store Location City Type",
            [
                "Tier 1",
                "Tier 2",
                "Tier 3"
            ]
        )

        store_type = st.selectbox(
            "Store Type",
            [
                "Food Mart",
                "Departmental Store",
                "Supermarket Type1",
                "Supermarket Type2"
            ]
        )

        store_age = st.number_input(
            "Store Age",
            min_value=0,
            value=20
        )

        product_id_code = st.selectbox(
            "Product ID Code",
            [
                "FD",
                "DR",
                "NC"
            ]
        )

    if st.button("Predict Sales"):

        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar_content,
            "Product_Allocated_Area": product_allocated_area,
            "Product_Type": product_type,
            "Product_MRP": product_mrp,
            "Store_Id": store_id,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_tier,
            "Store_Type": store_type,
            "Store_Age": store_age,
            "Product_ID_Code": product_id_code
        }

        try:

            response = requests.post(
                f"{BACKEND_URL}/v1/predict",
                json=payload
            )

            result = response.json()

            st.success(
                f"Predicted Sales Revenue: ${result['predicted_sales']:,.2f}"
            )

        except Exception as e:

            st.error(str(e))


# ==========================================================
# BATCH PREDICTION
# ==========================================================

with tab2:

    st.header("Batch Sales Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        st.subheader("Input File Preview")

        preview_df = pd.read_csv(uploaded_file)

        st.dataframe(preview_df.head())

        uploaded_file.seek(0)

        if st.button("Generate Predictions"):

            files = {
                "file": uploaded_file
            }

            try:

                response = requests.post(
                    f"{BACKEND_URL}/v1/predictbatch",
                    files=files
                )

                results = response.json()

                prediction_df = preview_df.copy()

                prediction_df["Predicted_Sales"] = (
                    results["predictions"]
                )

                st.subheader("Prediction Results")

                st.dataframe(prediction_df)

                csv = prediction_df.to_csv(index=False)

                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="superkart_predictions.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(str(e))
