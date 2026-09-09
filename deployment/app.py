
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# --- Configuration --- #
MODEL_REPO_ID = "senthil31/tourism-product-prediction-tuned-model"
MODEL_FILENAME = "model.pkl"
FEATURES_FILENAME = "model_features.pkl"

# --- Load Model and Features --- #
@st.cache_resource
def load_model_from_hub():
    try:
        model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        return None

@st.cache_resource
def load_features_from_hub():
    try:
        features_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=FEATURES_FILENAME)
        features = joblib.load(features_path)
        return features
    except Exception as e:
        st.error(f"Error loading features from Hugging Face Hub: {e}")
        return None

model = load_model_from_hub()
model_features = load_features_from_hub()

if model is None or model_features is None:
    st.stop() # Stop if model or features could not be loaded

# --- Streamlit UI --- #
st.set_page_config(page_title="Tourism Product Prediction", layout="centered")
st.title("Tourism Product Purchase Predictor ")
st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# --- Input Form --- #
with st.form("prediction_form"):
    st.header("Customer Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=99, value=30)
        typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
        citytier = st.selectbox("City Tier", [1, 2, 3])
        durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=1.0, value=8.0)
        numberofpersonvisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
        numberoffollowups = st.number_input("Number of Follow-ups", min_value=0, value=3)
        preferredpropertystar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

    with col2:
        numberoftrips = st.number_input("Number of Trips Annually", min_value=0.0, value=1.0)
        passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        pitchsatisfactionscore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
        owncar = st.selectbox("Owns a Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        numberofchildrenvisiting = st.number_input("Number of Children Visiting (under 5)", min_value=0.0, value=0.0)
        monthlyincome = st.number_input("Monthly Income", min_value=0.0, value=25000.0)
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        designation = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP", "Director"])

    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Create input DataFrame
        input_data = {
            'Age': age,
            'TypeofContact': typeofcontact,
            'CityTier': citytier,
            'DurationOfPitch': durationofpitch,
            'NumberOfPersonVisiting': numberofpersonvisiting,
            'NumberOfFollowups': numberoffollowups,
            'PreferredPropertyStar': preferredpropertystar,
            'MaritalStatus': maritalstatus,
            'NumberOfTrips': numberoftrips,
            'Passport': passport,
            'PitchSatisfactionScore': pitchsatisfactionscore,
            'OwnCar': owncar,
            'NumberOfChildrenVisiting': numberofchildrenvisiting,
            'MonthlyIncome': monthlyincome,
            'Occupation': occupation,
            'Gender': gender,
            'ProductPitched': productpitched,
            'Designation': designation
        }
        input_df = pd.DataFrame([input_data])

        # One-hot encode categorical features
        input_df_processed = pd.get_dummies(input_df, drop_first=True)

        # Align columns with model_features (from training data)
        # Add missing columns with 0 and reorder
        final_input_df = pd.DataFrame(columns=model_features, index=input_df_processed.index)
        for col in model_features:
            if col in input_df_processed.columns:
                final_input_df[col] = input_df_processed[col]
            else:
                final_input_df[col] = 0

        # Ensure all columns are numeric (e.g., for StandardScaler in pipeline)
        for col in final_input_df.columns:
            if final_input_df[col].dtype == 'bool': # Convert boolean columns to int
                final_input_df[col] = final_input_df[col].astype(int)

        # Make prediction
        try:
            prediction = model.predict(final_input_df)[0]
            prediction_proba = model.predict_proba(final_input_df)[:, 1][0]

            st.subheader("Prediction Result")
            if prediction == 1:
                st.success(f"The customer is likely to purchase the Wellness Tourism Package (Probability: {prediction_proba:.2f})")
            else:
                st.info(f"The customer is not likely to purchase the Wellness Tourism Package (Probability: {prediction_proba:.2f})")

            st.write(f"*Probability of purchase: {prediction_proba:.2f}* (1 = Yes, 0 = No)")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

