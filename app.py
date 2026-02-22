import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Student Performance Dashboard",
    layout="wide"
)

st.title("🎓 Student Performance Prediction Dashboard")
st.markdown("Interactive dashboard for prediction and performance analysis")

# ---------------------------------
# LOAD DATA & MODEL (PATH FIXED)
# ---------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/student_exam_scores.csv")

df = load_data()

model = joblib.load("models/student_performance_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ---------------------------------
# SIDEBAR: USER INPUTS
# ---------------------------------
st.sidebar.header("📌 Student Inputs")

hours_studied = st.sidebar.slider("Study Hours", 0, 12, 5)
sleep_hours = st.sidebar.slider("Sleep Hours", 3, 10, 7)
attendance_percent = st.sidebar.slider("Attendance (%)", 50, 100, 85)
previous_scores = st.sidebar.slider("Previous Score", 40, 100, 75)

# ---------------------------------
# PREDICTION
# ---------------------------------
input_data = np.array([
    hours_studied,
    sleep_hours,
    attendance_percent,
    previous_scores
]).reshape(1, -1)

input_scaled = scaler.transform(input_data)
prediction = model.predict(input_scaled)[0]

st.sidebar.markdown("---")
st.sidebar.metric(
    "📊 Predicted Exam Score",
    f"{prediction:.2f}"
)

# ---------------------------------
# MAIN DASHBOARD TABS
# ---------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Dataset Overview",
    "📊 Visual Analysis",
    "📉 Feature Relationships",
    "📌 Student Profile"
])

# ---------------------------------
# TAB 1: DATASET OVERVIEW
# ---------------------------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

# ---------------------------------
# TAB 2: VISUAL ANALYSIS
# ---------------------------------
with tab2:
    st.subheader("Feature Distributions")

    fig, ax = plt.subplots(figsize=(12,6))
    df.hist(bins=20, ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ---------------------------------
# TAB 3: FEATURE RELATIONSHIPS
# ---------------------------------
with tab3:
    st.subheader("Study Hours vs Exam Score")

    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(
        x="hours_studied",
        y="exam_score",
        hue="attendance_percent",
        data=df,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Impact of Sleep Hours on Exam Score")

    fig, ax = plt.subplots(figsize=(7,5))
    sns.lineplot(
        x="sleep_hours",
        y="exam_score",
        data=df,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("3D Relationship: Study, Sleep & Exam Score")

    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        df["hours_studied"],
        df["sleep_hours"],
        df["exam_score"],
        c=df["exam_score"],
        cmap="viridis"
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Sleep Hours")
    ax.set_zlabel("Exam Score")

    st.pyplot(fig)

# ---------------------------------
# TAB 4: STUDENT PROFILE (RADAR)
# ---------------------------------
with tab4:
    st.subheader("Average Student Profile (Normalized Radar Chart)")

    features = [
        "hours_studied",
        "sleep_hours",
        "attendance_percent",
        "previous_scores"
    ]

    scaler_radar = MinMaxScaler()
    scaled = scaler_radar.fit_transform(df[features])
    avg_profile = scaled.mean(axis=0)

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False)
    avg_profile = np.concatenate([avg_profile, [avg_profile[0]]])
    angles = np.concatenate([angles, [angles[0]]])

    fig = plt.figure(figsize=(6,6))
    plt.polar(angles, avg_profile)
    plt.fill(angles, avg_profile, alpha=0.3)
    plt.thetagrids(angles[:-1] * 180/np.pi, features)
    plt.title("Average Student Profile")

    st.pyplot(fig)

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("---")
st.markdown("📌 End-to-end ML Dashboard | Student Performance Prediction")