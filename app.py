import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Risk Assessment", page_icon="🩺", layout="wide")

st.title("🩺 Diabetes Risk Assessment System")
st.markdown("### AI-based Clinical Screening Tool")

st.markdown("---")

# Patient info
st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Patient Name")

with col2:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.markdown("---")

st.subheader("🧪 Clinical Parameters")

c1, c2, c3 = st.columns(3)

with c1:
    preg = st.number_input("Pregnancies", 0, 20, 0)
    glu = st.number_input("Glucose Level", 0, 250, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)

with c2:
    skin = st.number_input("Skin Thickness", 0, 100, 20)
    ins = st.number_input("Insulin Level", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)

with c3:
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

st.markdown("---")

if st.button("🔍 Generate Prediction"):

    with st.spinner("Analyzing patient data..."):

        data = {
            "Pregnancies": preg,
            "Glucose": glu,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": ins,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
        }

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data
        )

        result = response.json()

    st.markdown("## 📄 Medical Report")

    st.write(f"**Patient Name:** {name}")
    st.write(f"**Age:** {age}")

    st.markdown("---")

    prob_non = result["non_diabetic_probability"] * 100
    prob_diab = result["diabetic_probability"] * 100

    col1, col2 = st.columns(2)

    col1.metric("Non-Diabetic Probability", f"{prob_non:.2f}%")
    col2.metric("Diabetic Probability", f"{prob_diab:.2f}%")

    st.progress(int(prob_diab))

    st.markdown("---")

    if result["prediction"] == 1:
        st.error("⚠ High Risk of Diabetes Detected")
        diagnosis = "High Risk of Diabetes"
    else:
        st.success("✅ Low Risk / No Diabetes Detected")
        diagnosis = "Low Risk"

    st.markdown("### 📝 Clinical Conclusion")

    st.write(f"""
    Based on the clinical parameters entered, the system predicts that the patient is classified as **{diagnosis}**.

    ⚠ This system is designed as a **screening support tool** and should not replace professional medical diagnosis.
    """)

    st.markdown("---")

    # Input Data Summary
    st.subheader("📊 Input Data Summary")
    df = pd.DataFrame([data])
    st.dataframe(df)

    # Feature chart
    features = [
        "Pregnancies","Glucose","BloodPressure",
        "SkinThickness","Insulin","BMI",
        "DiabetesPedigreeFunction","Age"
    ]

    values = [preg,glu,bp,skin,ins,bmi,dpf,age]

    st.subheader("📊 Patient Feature Overview")

    fig, ax = plt.subplots()
    ax.barh(features, values)
    st.pyplot(fig)

    # Risk level
    st.subheader("⚠ Risk Level")

    risk = prob_diab

    if risk < 30:
        st.success("Low Risk")
    elif risk < 60:
        st.warning("Moderate Risk")
    else:
        st.error("High Risk")