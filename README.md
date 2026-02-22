# 🎓 Student Performance Prediction System

## 📌 Project Overview
The Student Performance Prediction System is an end-to-end machine learning project designed to predict students’ final exam scores based on academic and lifestyle factors.  
The system uses regression models to analyze patterns in student data and provides predictions along with interactive visual insights through a dashboard.

This project follows the complete Machine Learning Life Cycle, including data analysis, model training, evaluation, deployment, and visualization.

---

## 🎯 Objective
- Predict students’ final exam scores using machine learning
- Analyze how factors like study hours, sleep hours, attendance, and previous scores affect performance
- Provide an interactive dashboard for predictions and data insights
- Help educators identify students who may need academic support

---
## 📊 Dataset Description
The dataset contains student-related academic and behavioral features:

- `hours_studied` – Number of hours studied per day  
- `sleep_hours` – Average sleep duration  
- `attendance_percent` – Attendance percentage  
- `previous_scores` – Previous academic performance  
- `exam_score` – Final exam score (target variable)

---

## 🔍 Machine Learning Pipeline

### 1. Data Collection
- Dataset loaded from a CSV file stored locally in the `data/` directory.

### 2. Data Preparation & Cleaning
- Checked for missing values and data types
- Removed non-predictive identifier columns
- Ensured only numerical features were used for modeling

### 3. Exploratory Data Analysis (EDA)
- Histograms to analyze feature distributions
- Correlation heatmaps to identify relationships
- Scatter plots and 3D plots for feature interaction analysis
- Radar chart for average student profile visualization

### 4. Feature Engineering
- Selected relevant features for prediction
- Defined input variables (X) and target variable (y)
- Applied feature scaling for linear models

### 5. Model Training
The following regression models were trained:
- Linear Regression  
- Lasso Regression  
- Ridge Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

### 6. Model Evaluation
Models were evaluated using:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

### 7. Best Model Selection
- **Ridge Regression** achieved the highest R² score and lowest error
- It was selected as the final model

### 8. Model Saving
- Final model and scaler were saved using `joblib`
- Stored inside the `models/` directory

---

## 🧠 Final Output
- The system predicts a student’s final exam score based on user input
- Outputs are numerical and easy to interpret
- The dashboard also provides visual insights into student performance patterns

---

## 🖥️ Frontend & Backend Details

### Frontend
- Built using **Streamlit**
- Interactive sliders for user input
- Tabs for:
  - Dataset overview
  - Visual analysis
  - Feature relationships
  - Student profile radar chart

### Backend
- Machine learning model loaded from `.pkl` file
- Scaler applied to user input before prediction
- Real-time prediction generated using the trained model

---

## 🚀 How to Run the Project

### 1. Clone or Download the Project
Make sure all files are in the same folder structure as shown above.

### 2. Install Dependencies
Run the following command in the project root directory:

```bash
pip install -r requirements.txt

```
---
## Run the Dashboard

### Start the Streamlit application using:

    streamlit run app.py
---

