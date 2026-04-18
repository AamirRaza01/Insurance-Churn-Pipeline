import os
import sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from insurance_predictor.pipeline.prediction_pipeline import PredictionPipeline, CustomData
from insurance_predictor.pipeline.training_pipeline import TrainingPipeline

st.set_page_config(
    page_title="Insurance Lead Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Vehicle Insurance Lead Predictor")
st.markdown("Predict whether a customer is interested in vehicle insurance.")
st.divider()

# Sidebar
st.sidebar.title("⚙️ Options")
action = st.sidebar.radio(
    "Choose Action",
    ["Make Prediction", "Train Model"]
)

if action == "Train Model":
    st.subheader("🔁 Retrain the Model")
    st.warning("This will run the full training pipeline. It may take a few minutes.")

    if st.button("Start Training"):
        with st.spinner("Training pipeline running..."):
            try:
                pipeline = TrainingPipeline()
                pipeline.run_pipeline()
                st.success("✅ Training pipeline completed successfully!")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

else:
    st.subheader("📋 Enter Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Age = st.number_input("Age", min_value=18, max_value=100, value=35)
        Driving_License = st.selectbox("Driving License", [1, 0],
                                        format_func=lambda x: "Yes" if x == 1 else "No")
        Region_Code = st.number_input("Region Code", min_value=0.0, max_value=60.0, value=28.0)
        Previously_Insured = st.selectbox("Previously Insured", [0, 1],
                                           format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        Vehicle_Age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
        Vehicle_Damage = st.selectbox("Vehicle Damage", ["Yes", "No"])
        Annual_Premium = st.number_input("Annual Premium", min_value=0.0, value=30000.0)
        Policy_Sales_Channel = st.number_input("Policy Sales Channel",
                                                min_value=0.0, max_value=200.0, value=26.0)
        Vintage = st.number_input("Vintage (days)", min_value=0, max_value=400, value=150)

    st.divider()

    if st.button("🔍 Predict", use_container_width=True):
        try:
            custom_data = CustomData(
                Gender=Gender,
                Age=Age,
                Driving_License=Driving_License,
                Region_Code=Region_Code,
                Previously_Insured=Previously_Insured,
                Vehicle_Age=Vehicle_Age,
                Vehicle_Damage=Vehicle_Damage,
                Annual_Premium=Annual_Premium,
                Policy_Sales_Channel=Policy_Sales_Channel,
                Vintage=Vintage
            )

            dataframe = custom_data.get_data_as_dataframe()
            pipeline = PredictionPipeline()
            prediction = pipeline.predict(dataframe)

            st.divider()
            if prediction[0] == 1:
                st.success("✅ This customer is **INTERESTED** in vehicle insurance!")
                st.balloons()
            else:
                st.error("❌ This customer is **NOT INTERESTED** in vehicle insurance.")

            st.subheader("📊 Input Summary")
            st.dataframe(dataframe)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.info("Please train the model first using the sidebar.")