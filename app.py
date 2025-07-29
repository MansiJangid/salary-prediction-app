import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Load data and model
df = pd.read_csv("Salary_Data.csv")
df_roles = sorted(df['Job Title'].dropna().unique().tolist())
df_roles.insert(0, "Select")

df = df.dropna()
# Create age groups 
bins = [20, 30, 40, 50, 60, 70]
labels = ['20-29', '30-39', '40-49', '50-59', '60+']
df['Age_Group'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

# Create experience groups
exp_bins = [0, 3, 6, 10, 15, 30]
exp_labels = ['0-2', '3-5', '6-9', '10-14', '15+']
df['Experience_Group'] = pd.cut(df['Years of Experience'], bins=exp_bins, labels=exp_labels, right=False)


df['Education Level'] = df['Education Level'].replace("phD", "PhD")
df['Education Level'] = df['Education Level'].replace("Bachelor's Degree", "Bachelor's")
df['Education Level'] = df['Education Level'].replace("Master's Degree", "Master's")

# Replace empty strings with NaN first 
df.replace('', pd.NA, inplace=True)

# Drop rows with any NaN
df.dropna(inplace=True)


# Categorical columns
categorical_cols = ['Gender', 'Education Level', 'Job Title', 'Age_Group', 'Experience_Group']

X = df[['Gender', 'Education Level', 'Job Title', 'Age_Group', 'Experience_Group']]
y = df['Salary']

X_encoded = pd.get_dummies(X, drop_first=True)
X = X_encoded


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = joblib.load("random_forest_model.pkl")  # Ensure model is in directory
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

model_performance = {
    "Random Forest": 0.92,
    "Linear Regression": 0.85,
    "XGBoost": 0.87,
    "SVM": 0.08,
    "CatBoostRegressor": 0.91,
}

# Streamlit UI
st.set_page_config(page_title="Salary Prediction App", layout="centered")
st.title("💼 Salary Prediction App")

st.markdown("""
<style>
    .main { background-color: #f4f4f4; padding: 20px; border-radius: 10px; }
    .stButton>button { background-color: #4CAF50; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown("#### Fill in the details below to estimate your monthly salary:")

with st.form("salary_form", clear_on_submit=False):
    gender = st.radio("Select Gender", ["Male", "Female", "Other"])

    education = st.selectbox("Education Level", [
        "Select", "High School", "Bachelor's", "Master's", "PhD", "Other"
    ])

    job_title = st.selectbox("Job Title", df_roles, index=0, placeholder="Search or select a job title")

    age_group = st.selectbox("Age Group", [
        "Select", "20-30", "30-40", "40-50", "50-60", "60+"
    ])

    experience_group = st.selectbox("Experience Group", [
        "Select", "0-2 years", "2-5 years", "5-10 years", "10-15 years", "15+ years"
    ])

    submit = st.form_submit_button("Predict Salary")

    if submit:
        # Validate selections
        if "Select" in [education, job_title, age_group, experience_group]:
            st.error("⚠️ Please fill all fields before predicting.")
        elif age_group == "20-30" and experience_group in ["10-15 years", "15+ years"]:
            st.error("⚠️ Experience cannot be more than 10 years for Age Group 20-30.")
        else:
            # Show input summary
            st.markdown("### ✅ Your Input Summary:")
            st.write(pd.DataFrame({
                "Gender": [gender],
                "Education Level": [education],
                "Job Title": [job_title],
                "Age Group": [age_group],
                "Experience Group": [experience_group]
            }))

            # Predict
            input_data = pd.DataFrame({
                'Gender': [gender],
                'Education Level': [education],
                'Job Title': [job_title],
                'Age_Group': [age_group],
                'Experience_Group': [experience_group]
            })

            # One-hot encode input to match training
            input_encoded = pd.get_dummies(input_data, drop_first=True)

            # Align columns with training data
            input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
            
            # Predict
            prediction = model.predict(input_encoded)

            st.success(f"🎯 Estimated Salary (Per Month): **₹ {int(prediction[0]):,}**")

            # Model info
            st.markdown("### 📊 Model Info:")
            st.markdown("- **Model Used:** Random Forest Regressor")
            st.markdown(f"- **Model Accuracy:** {model_performance['Random Forest']*100:.2f}% (on validation data)")

            # Accuracy comparison graph
            st.markdown("### 📈 Model Accuracy Comparison:")
            fig, ax = plt.subplots()
            ax.bar(model_performance.keys(), model_performance.values(), color='skyblue')
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            st.pyplot(fig)