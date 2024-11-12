import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the saved logistic regression model
model = joblib.load('logistic_regression_model.pkl')

# # Add a title
st.title("Customer Churn Prediction")

# Add a description
st.text("Using Machine-Learning Algorithm to predict if a customer would churn or not")

# Add an image
st.image('customer-churn1.jpg')



# Define the categorical features
categories = ['gender', 'Partner', 'Dependents',
              'PhoneService', 'MultipleLines', 'InternetService',
              'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
              'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
              'PaymentMethod', 'TotalCharges']


# Create a function to collect user inputs
def get_user_inputs():
    user_input = {}

    # Using Streamlit components to collect input
    user_input['gender'] = st.selectbox('Gender', ['Select', 'Male', 'Female'])
    user_input['Partner'] = st.selectbox('Partner', ['Select','Yes', 'No'])
    user_input['Dependents'] = st.selectbox('Dependents', ['Select','Yes', 'No'])
    user_input['PhoneService'] = st.selectbox('Phone Service', ['Select','Yes', 'No'])
    user_input['MultipleLines'] = st.selectbox('Multiple Lines', ['Select','Yes', 'No'])
    user_input['InternetService'] = st.selectbox('Internet Service', ['Select','DSL', 'Fiber optic', 'No'])
    user_input['OnlineSecurity'] = st.selectbox('Online Security', ['Select', 'Yes', 'No'])
    user_input['OnlineBackup'] = st.selectbox('Online Backup', ['Select', 'Yes', 'No'])
    user_input['DeviceProtection'] = st.selectbox('Device Protection', ['Select', 'Yes', 'No'])
    user_input['TechSupport'] = st.selectbox('Tech Support', ['Select', 'Yes', 'No'])
    user_input['StreamingTV'] = st.selectbox('Streaming TV', ['Select', 'Yes', 'No'])
    user_input['StreamingMovies'] = st.selectbox('Streaming Movies', ['Select', 'Yes', 'No'])
    user_input['Contract'] = st.selectbox('Contract', ['Select', 'Month-to-month', 'One year', 'Two year'])
    user_input['PaperlessBilling'] = st.selectbox('Paperless Billing', ['Select', 'Yes', 'No'])
    user_input['PaymentMethod'] = st.selectbox('Payment Method',
                                               ['Select', 'Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'])
    user_input['TotalCharges'] = st.number_input('Total Charges')  # Ensure it's numeric

    return user_input


# Collect user inputs
user_input = get_user_inputs()

# Convert user inputs into a DataFrame
user_input_df = pd.DataFrame([user_input])

# Encoding the categorical features
encoder = LabelEncoder()

# Encoding only for the relevant categorical columns (except TotalCharges)
for feature in categories[:-1]:
    user_input_df[feature] = encoder.fit_transform(user_input_df[feature])

# Convert TotalCharges to numeric
user_input_df['TotalCharges'] = pd.to_numeric(user_input_df['TotalCharges'], errors='coerce')

# Add an image
st.image('customer-churn1.jpg')

# Make predictions
if st.button('Predict'):
    prediction = model.predict(user_input_df)

    # Display result
    if prediction[0] == 1:
        st.success("Yes, this customer will churn.")
    else:
        st.success("No this customer will not churn.")