import streamlit as st
import pandas as pd
import pickle
import os

# Configure the page layout to be wide.
st.set_page_config(layout="wide")

# ------------------------------------------------------------
# Function used by the pipeline to extract the year_of_birth from customerdob.
# This is required for the model pipeline to run predictions successfully.
# ------------------------------------------------------------
def extract_year_of_birth(X):
    X = X.copy()
    X["customerdob"] = pd.to_datetime(X["customerdob"], errors='coerce', format="%Y")
    X["year_of_birth"] = X["customerdob"].dt.year
    return X.drop(columns=["customerdob"])

# ------------------------------------------------------------
# Load the pre-trained pipeline from a local pickle file.
# The pipeline contains both preprocessing steps and the trained model.
# ------------------------------------------------------------
with open("model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Set the application title and a brief instruction
st.title("Insurance Policy Price Estimator")
st.write("Provide customer details to estimate the yearly premium.")

# ------------------------------------------------------------
# Define original code values and mappings for user-friendly displays.
# The user will see descriptive names, and these will be mapped back to
# original codes before passing to the model for prediction.
# ------------------------------------------------------------

# Original code lists
original_sex = ['M', 'F']
original_smokingclass = ['S', 'N']
original_maritalstatus = ['single', 'married', 'divorced']
original_prodcode = ['A0001', 'A0002', 'A0003', 'A0004']
original_issuestate = [
    'ND', 'MN', 'PA', 'MD', 'FL', 'NV', 'WI', 'UT', 'MT', 'MS',
    'DE', 'MI', 'SD', 'CT', 'NM', 'AL', 'NE', 'OH', 'IL', 'MA',
    'AK', 'AR', 'GA', 'IN', 'CA', 'NH', 'OR', 'CO', 'KS', 'NC',
    'ME', 'NJ', 'AZ', 'HI', 'WA', 'WV', 'ID', 'VA', 'TX', 'LA',
    'KY', 'RI', 'TN', 'WY', 'MO', 'SC', 'IA', 'OK', 'NY', 'VT', 'DC'
]

# User-friendly mapping for sex
sex_map = {
    'M': "Male",
    'F': "Female"
}
name_to_sex_code = {v: k for k, v in sex_map.items()}
sex_names = list(sex_map.values())

# User-friendly mapping for smoking class
smoking_map = {
    'S': "Smoker",
    'N': "Non-smoker"
}
name_to_smoking_code = {v: k for k, v in smoking_map.items()}
smoking_names = list(smoking_map.values())

# User-friendly mapping for product codes
product_map = {
    'A0001': "Basic Life Insurance",
    'A0002': "Premium Life Insurance",
    'A0003': "Family Life Insurance",
    'A0004': "Senior Life Insurance"
}
name_to_code = {v: k for k, v in product_map.items()}
product_names = list(product_map.values())

# Marital status and issue state remain as-is without special mapping
maritalstatus_values = original_maritalstatus
issuestate_values = original_issuestate

# ------------------------------------------------------------
# Create a form to collect user input.
# Using columns to place two fields in one row for a more compact layout.
# ------------------------------------------------------------
with st.form(key='insurance_form'):
    # First row: Sex and Year of Birth
    col1, col2 = st.columns(2)
    with col1:
        selected_sex_name = st.selectbox("Sex", sex_names)
    with col2:
        customeryear = st.number_input("Year of Birth", min_value=1960, max_value=2015, value=1980)

    # Second row: Smoking Class and Marital Status
    col3, col4 = st.columns(2)
    with col3:
        selected_smoking_name = st.selectbox("Smoking Class", smoking_names)
    with col4:
        maritalstatus = st.selectbox("Marital Status", maritalstatus_values)

    # Third row: Coverage Unit and Policy Term
    col5, col6 = st.columns(2)
    with col5:
        coverageunit = st.number_input("Coverage Unit", min_value=1, value=96)
    with col6:
        policyterm = st.number_input("Policy Term (years)", min_value=1, value=15)

    # Fourth row: Insurance Product and Issue State
    col7, col8 = st.columns(2)
    with col7:
        selected_product_name = st.selectbox("Insurance Product", product_names)
    with col8:
        issuestate = st.selectbox("Issue State", issuestate_values)

    # Submit button to trigger prediction
    submit_button = st.form_submit_button(label="Estimate Quote")

# ------------------------------------------------------------
# On form submission, convert user-friendly names back to original codes,
# prepare the input DataFrame, and run the model prediction.
# ------------------------------------------------------------
if submit_button:
    sex_code = name_to_sex_code[selected_sex_name]
    smoking_code = name_to_smoking_code[selected_smoking_name]
    prodcode = name_to_code[selected_product_name]

    input_data = {
        "sex": [sex_code],
        "customerdob": [str(customeryear)],
        "smokingclass": [smoking_code],
        "maritalstatus": [maritalstatus],
        "coverageunit": [coverageunit],
        "policyterm": [policyterm],
        "prodcode": [prodcode],
        "issuestate": [issuestate]
    }

    input_df = pd.DataFrame(input_data)

    # Perform prediction using the loaded pipeline
    prediction = pipeline.predict(input_df)[0]

    # Display the prediction result
    st.success(f"Estimated Yearly Policy Price: {prediction:.2f}")
