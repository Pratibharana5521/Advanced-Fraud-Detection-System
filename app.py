import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


model = joblib.load('fraud_detection_rf_model (1).pkl')  


transaction_type_encoder = LabelEncoder()
transaction_type_encoder.fit(['Credit', 'Debit'])

location_encoder = LabelEncoder()
location_encoder.fit(['New York', 'Los Angeles', 'Chicago', 'Miami', 'Dallas'])  

st.title("💳 Fraud Detection System")
st.write("Enter transaction details to check if it's fraudulent.")


transaction_amount = st.number_input("Transaction Amount", min_value=0.0, step=1.0)

transaction_type = st.selectbox("Transaction Type", ['Credit', 'Debit'])
location = st.selectbox("Transaction Location", ['New York', 'Los Angeles', 'Chicago', 'Miami', 'Dallas'])


if st.button("Detect Fraud"):
    encoded_type = transaction_type_encoder.transform([transaction_type])[0]
    encoded_location = location_encoder.transform([location])[0]

    
    input_df = pd.DataFrame({
        'transaction_amount': [transaction_amount],
        'transaction_type': [encoded_type],
        'location': [encoded_location]
    })

    
    prediction = model.predict(input_df)[0]
    result = "🚫 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
    color = "red" if prediction == 1 else "green"

    st.markdown(f"<h3 style='color:{color}'>{result}</h3>", unsafe_allow_html=True)
