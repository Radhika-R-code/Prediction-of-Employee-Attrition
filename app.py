import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("Employee Attrition Prediction System")

# Load model artifacts
model = joblib.load("decision_tree_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# Load dataset
df = pd.read_csv("Employee-Attrition.csv")

# Create a sub dataframe (similar dataset version used for modeling purposes)
col_keep = [
    "Attrition", "Age", "Gender", "MaritalStatus", "DistanceFromHome",
    "Education", "EducationField", "Department", "JobRole", "JobLevel",
    "BusinessTravel", "OverTime", "MonthlyIncome", "StockOptionLevel",
    "JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
    "WorkLifeBalance", "JobInvolvement", "YearsAtCompany",
    "YearsSinceLastPromotion", "NumCompaniesWorked"
]
df_sub = df[col_keep].copy()

st.success("Loaded df_sub successfully!")
st.write("df_sub shape:", df_sub.shape)
st.dataframe(df_sub.head())

# Create Tabs 
tab1, tab2, tab3 = st.tabs(["Dashboard", "Single prediction", "Batch prediction"])


# TAB 1: Dashboard

with tab1:
    st.markdown("## Workforce Overview")

    total_employees = df_sub.shape[0]
    attrition_count = (df_sub["Attrition"] == "Yes").sum()
    attrition_rate = (attrition_count / total_employees) * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Employees", int(total_employees))
    with c2:
        st.metric("Employees Left", int(attrition_count))
    with c3:
        st.metric("Attrition Rate (%)", f"{attrition_rate:.2f}")

    st.markdown("## Attrition Insights")

    # Attrition Distribution
    st.subheader("Attrition Distribution")
    st.bar_chart(df_sub["Attrition"].value_counts())
    st.caption("Most employees stayed, so attrition is imbalanced (more 'No' than 'Yes').")

    # Attrition by Department
    st.subheader("Attrition by Department")
    dept_attrition = (
        df_sub.groupby("Department")["Attrition"]
        .value_counts()
        .unstack()
        .fillna(0)
    )
    st.bar_chart(dept_attrition)
    st.caption("Departments with higher attrition may need targeted retention strategies.")

    # Attrition vs OverTime
    st.subheader("Attrition vs OverTime")
    ot_attrition = (
        df_sub.groupby("OverTime")["Attrition"]
        .value_counts()
        .unstack()
        .fillna(0)
    )
    st.bar_chart(ot_attrition)
    st.caption("OverTime is a strong signal: overtime workers often show higher attrition risk.")

    st.subheader("Preview of cleaned dataset (df_sub)")
    st.dataframe(df_sub.head(15), use_container_width=True)


# TAB 2: Single prediction
with tab2:
    st.markdown("## Predict Attrition for One Employee")
    st.caption("Inputs match the same features used during training. The model outputs attrition probability.")

    # Demographics
    age = st.number_input("Age", 18, 60, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    distance = st.number_input("Distance From Home", 0, 60, 10)
    education = st.selectbox("Education Level", [1, 2, 3, 4, 5])
    edu_field = st.selectbox(
        "Education Field",
        ["Life Sciences", "Medical", "Marketing", "Technical Degree",
         "Human Resources", "Other"]
    )

    # Job & Organization
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.selectbox(
        "Job Role",
        ["Sales Executive", "Research Scientist", "Laboratory Technician",
         "Manufacturing Director", "Healthcare Representative",
         "Manager", "Sales Representative", "Research Director", "Human Resources"]
    )
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    overtime = st.selectbox("OverTime", ["Yes", "No"])

    # Compensation
    monthly_income = st.number_input("Monthly Income", 1000, 300000, 5000)
    stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])

    # Satisfaction & Engagement
    job_sat = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
    env_sat = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
    rel_sat = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
    wlb = st.selectbox("Work Life Balance", [1, 2, 3, 4])
    job_inv = st.selectbox("Job Involvement", [1, 2, 3, 4])

    # Tenure & Career Growth
    years_company = st.number_input("Years at Company", 0, 40, 5)
    years_promo = st.number_input("Years Since Last Promotion", 0, 20, 1)
    num_companies = st.number_input("Number of Companies Worked", 0, 20, 2)

    threshold = st.slider("Decision Threshold (risk cut-off)", 0.10, 0.90, 0.50, 0.05)

    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "MaritalStatus": marital,
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": edu_field,
        "Department": department,
        "JobRole": job_role,
        "JobLevel": job_level,
        "BusinessTravel": business_travel,
        "OverTime": overtime,
        "MonthlyIncome": monthly_income,
        "StockOptionLevel": stock_option,
        "JobSatisfaction": job_sat,
        "EnvironmentSatisfaction": env_sat,
        "RelationshipSatisfaction": rel_sat,
        "WorkLifeBalance": wlb,
        "JobInvolvement": job_inv,
        "YearsAtCompany": years_company,
        "YearsSinceLastPromotion": years_promo,
        "NumCompaniesWorked": num_companies
    }])

    if st.button("Predict Attrition"):
        try:
            X_input = preprocessor.transform(input_df)
            prob = model.predict_proba(X_input)[0][1]
            pred = "Yes" if prob >= threshold else "No"

            st.subheader("Prediction Result")
            st.write(f"Probability of Attrition: {prob:.2f}")

            if pred == "Yes":
                st.error("Attrition Risk: YES (Employee is likely to leave)")
            else:
                st.success("Attrition Risk: NO (Employee is likely to stay)")

        except Exception as e:
            st.error("Prediction failed. Check preprocessing/model compatibility.")
            st.exception(e)


# TAB 3: Batch prediction

with tab3:
    st.markdown("## Batch Prediction (Upload CSV)")
    st.caption("Upload a CSV containing the same columns as model inputs (df_sub without Attrition).")

    expected_cols = [c for c in col_keep if c != "Attrition"]
    st.markdown("### Expected Columns")
    st.code(", ".join(expected_cols))

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)

        missing = [c for c in expected_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        batch_input = batch_df[expected_cols].copy()

        Xb = preprocessor.transform(batch_input)
        probs = model.predict_proba(Xb)[:, 1]
        preds = ["Yes" if p >= 0.50 else "No" for p in probs]

        out = batch_df.copy()
        out["Attrition_Probability"] = probs
        out["Attrition_Prediction"] = preds

        st.success("Batch prediction completed!")
        st.dataframe(out.head(25), use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results CSV",
            data=csv_bytes,
            file_name="attrition_predictions.csv",
            mime="text/csv"
        )
