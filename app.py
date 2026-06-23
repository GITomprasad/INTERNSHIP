import streamlit as st
from main import predict_category, best_model

st.title("Medical Report Classification")
report = st.text_area("Enter medical report:")
if st.button("Predict"):
    category = predict_category(best_model, report)
    st.write(f"Predicted Category: {category}")
