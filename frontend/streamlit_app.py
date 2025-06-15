import streamlit as st
import requests

st.title("🛰️ Agricultural Land Suitability Detector")
st.write("Upload a satellite or drone image of your land plot:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)

    # Send to Flask backend
    files = {"file": uploaded_file}
    with st.spinner("Analyzing land..."):
        response = requests.post("http://localhost:5000/predict", files=files)

    if response.status_code == 200:
        data = response.json()
        predicted_class = data["class"]
        readable_result = data["result"]

        st.success(readable_result)
    else:
        st.error("Prediction failed. Try again.")

