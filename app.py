import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import shap
import joblib

# Load models
model = joblib.load("models/depression_model.pkl")
scaler = joblib.load("models/scaler.pkl")
le_gender = joblib.load("models/gender_encoder.pkl")
le_dept = joblib.load("models/department_encoder.pkl")
feature_names = joblib.load("models/feature_names.pkl")

st.title("Student Depression Prediction")

# Inputs
age = st.number_input("Age", min_value=16, max_value=40, step=1)
gender = st.selectbox("Gender", ["Male", "Female"])
department = st.selectbox("Department", list(le_dept.classes_))

cgpa = st.number_input("CGPA", 0.0, 10.0, step=0.1)
sleep = st.number_input("Sleep Duration (hours)", 0.0, 12.0, step=0.1)
study = st.number_input("Study Hours", 0.0, 12.0, step=0.1)
social = st.number_input("Social Media Hours", 0.0, 10.0, step=0.1)
activity = st.number_input("Physical Activity (minutes/week)", 0, 300, step=10)
stress = st.slider("Stress Level", 1, 10)

# Encode inputs
gender = le_gender.transform([gender])[0]
department = le_dept.transform([department])[0]

if st.button("Predict"):
    # Create DataFrame
    data = pd.DataFrame(
        [[age, gender, department, cgpa, sleep, study, social, activity, stress]],
        columns=feature_names
    )

    # Scale
    data_scaled = scaler.transform(data)

    # Prediction
    prob = model.predict_proba(data_scaled)[0][1]

    # SHAP Explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_scaled)

    st.write(f"Depression Probability: {prob*100:.2f}%")
    st.write("### Model Explanation")

    # --- FIXED SHAP HANDLING ---
    if isinstance(shap_values, list):
        impact_values = shap_values[1]  # class 1
    else:
        impact_values = shap_values

    impact_values = np.array(impact_values)

    # Ensure correct shape
    if impact_values.ndim == 2:
        impact_values = impact_values[0]

    impact_values = impact_values.flatten()

    # Safety check (prevents crash)
    if len(impact_values) != len(feature_names):
        min_len = min(len(impact_values), len(feature_names))
        impact_values = impact_values[:min_len]
        feature_names_trimmed = feature_names[:min_len]
    else:
        feature_names_trimmed = feature_names

    # Create DataFrame
    shap_df = pd.DataFrame({
        "Feature": feature_names_trimmed,
        "Impact": impact_values
    })

    # Sort by importance
    shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)
    
    # Top features table
    with st.expander("View detailed feature contributions"):
        st.dataframe(shap_df)

    # Bar chart visualization
    st.write("### Feature Impact Visualization")
    # Colors
    colors = ["red" if val > 0 else "green" for val in shap_df["Impact"]]

    fig, ax = plt.subplots()

    ax.bar(
        shap_df["Feature"],
        shap_df["Impact"],
        color=colors
    )

    #LEGEND 
    legend_elements = [
        Patch(facecolor='red', label='Increases Risk'),
        Patch(facecolor='green', label='Decreases Risk')
    ]

    ax.legend(handles=legend_elements)

    # Styling
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Impact on Prediction")
    plt.title("Feature Impact (SHAP Values)")

    st.pyplot(fig)


    # Risk Output
    if prob > 0.4:
        st.error("High Risk of Depression")
    else:
        st.success("Low Risk of Depression")

    # Top 3 factors
    st.write("### Key Factors Influencing Prediction")

    top_n = min(3, len(shap_df))

    for i in range(top_n):
        feature = shap_df.iloc[i]["Feature"]
        impact = shap_df.iloc[i]["Impact"]

        if impact > 0:
            st.write(f"🔺 {feature} increases risk")
        else:
            st.write(f"🔻 {feature} decreases risk")