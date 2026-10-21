# Health Risk Prediction System — Google Colab / Python notebook source
# Educational demo using synthetic data only; not medical advice.

# # 🩺 Health Risk Prediction System
# ### Advanced Python + Machine Learning + Data Analytics (Google Colab)
# 
# **Educational demo only:** this notebook uses synthetic data and simplified illustrative labels. It is not a medical device, does not diagnose disease, and must not be used for treatment or emergency decisions.
# 
# Features:
# - Synthetic dataset generation
# - Exploratory data analysis and charts
# - Random Forest classification
# - Holdout evaluation and confusion matrix
# - Interactive health-metrics form
# - Demo category prediction
# - CSV report download

# Cell 1 — Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from IPython.display import display, Markdown

print("Libraries imported successfully!")


# Cell 2 — Create synthetic educational dataset
# These are artificially generated records, not real patient data.

def make_synthetic_dataset(n_rows=3000, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 91, n_rows)
    bmi = np.clip(rng.normal(26, 5.2, n_rows), 15, 48).round(1)
    systolic_bp = np.clip(
        rng.normal(122 + (age - 45) * 0.25, 16, n_rows), 85, 220
    ).round().astype(int)
    diastolic_bp = np.clip(
        rng.normal(79 + (age - 45) * 0.08, 10, n_rows), 45, 135
    ).round().astype(int)
    glucose = np.clip(
        rng.normal(98 + (bmi - 25) * 1.2, 22, n_rows), 55, 300
    ).round().astype(int)
    heart_rate = np.clip(rng.normal(74, 12, n_rows), 42, 145).round().astype(int)
    cholesterol = np.clip(
        rng.normal(185 + (bmi - 25) * 1.5, 34, n_rows), 100, 360
    ).round().astype(int)
    activity_minutes = np.clip(
        rng.normal(35, 22, n_rows), 0, 180
    ).round().astype(int)

    df = pd.DataFrame({
        "age": age,
        "bmi": bmi,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "glucose": glucose,
        "heart_rate": heart_rate,
        "cholesterol": cholesterol,
        "activity_minutes": activity_minutes,
    })

    def illustrative_label(row):
        score = 0
        score += int(row["age"] >= 60)
        score += int(row["bmi"] >= 30)
        score += int(row["systolic_bp"] >= 140 or row["diastolic_bp"] >= 90)
        score += int(row["glucose"] >= 126)
        score += int(row["cholesterol"] >= 240)
        score += int(row["heart_rate"] >= 100)
        score += int(row["activity_minutes"] < 15)

        if score <= 1:
            return "Low"
        elif score <= 3:
            return "Moderate"
        return "High"

    df["demo_risk_category"] = df.apply(illustrative_label, axis=1)
    return df

df = make_synthetic_dataset()
print("Dataset shape:", df.shape)
display(df.head())


# Cell 3 — Dataset overview
print("Dataset information:")
display(df.describe().round(2))

print("\\nMissing values in each column:")
display(df.isnull().sum().to_frame("Missing values"))

print("\\nDemo category counts:")
display(df["demo_risk_category"].value_counts().to_frame("Count"))


# Cell 4 — Exploratory data analysis charts
sns.set_theme()

plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="demo_risk_category", order=["Low", "Moderate", "High"])
plt.title("Synthetic Demo Category Distribution")
plt.xlabel("Illustrative category")
plt.ylabel("Number of records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df.sample(min(700, len(df)), random_state=42),
    x="age", y="bmi", hue="demo_risk_category",
    hue_order=["Low", "Moderate", "High"], alpha=0.75
)
plt.title("Age vs BMI — Synthetic Records")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.drop(columns="demo_risk_category").corr(), annot=True, fmt=".2f", cmap="Blues")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()


# Cell 5 — Prepare features and split data
FEATURES = [
    "age", "bmi", "systolic_bp", "diastolic_bp",
    "glucose", "heart_rate", "cholesterol", "activity_minutes"
]
TARGET = "demo_risk_category"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# Cell 6 — Train the Random Forest model
model = RandomForestClassifier(
    n_estimators=180,
    max_depth=10,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Holdout accuracy on synthetic labels: {accuracy:.2%}")
print("\\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))
print("Reminder: this score measures reproduction of artificial labels, not real-world medical accuracy.")


# Cell 7 — Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    labels=["Low", "Moderate", "High"],
    cmap="Blues",
    values_format="d"
)
plt.title("Confusion Matrix — Synthetic Holdout Data")
plt.tight_layout()
plt.show()


# Cell 8 — Feature importance
importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

display(importance)

plt.figure(figsize=(8, 5))
sns.barplot(data=importance, x="Importance", y="Feature")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()


# Cell 9 — Interactive assessment form
# Enter fictional/demo values only. The output is not a medical assessment.

import ipywidgets as widgets
from IPython.display import display, clear_output

age_input = widgets.IntSlider(value=35, min=18, max=100, description="Age")
bmi_input = widgets.FloatSlider(value=24.0, min=10, max=60, step=0.1, description="BMI")
sys_input = widgets.IntSlider(value=118, min=70, max=240, description="Systolic BP")
dia_input = widgets.IntSlider(value=78, min=40, max=150, description="Diastolic BP")
glucose_input = widgets.IntSlider(value=95, min=40, max=500, description="Glucose")
hr_input = widgets.IntSlider(value=72, min=30, max=220, description="Heart rate")
chol_input = widgets.IntSlider(value=175, min=80, max=500, description="Cholesterol")
activity_input = widgets.IntSlider(value=30, min=0, max=600, description="Activity min/day")

analyze_button = widgets.Button(description="Analyze demo metrics", button_style="primary")
result_output = widgets.Output()

def run_prediction(_):
    with result_output:
        clear_output()
        sample = pd.DataFrame([{
            "age": age_input.value,
            "bmi": bmi_input.value,
            "systolic_bp": sys_input.value,
            "diastolic_bp": dia_input.value,
            "glucose": glucose_input.value,
            "heart_rate": hr_input.value,
            "cholesterol": chol_input.value,
            "activity_minutes": activity_input.value,
        }], columns=FEATURES)

        predicted_category = model.predict(sample)[0]
        probabilities = model.predict_proba(sample)[0]
        probability_map = dict(zip(model.classes_, probabilities))

        print("EDUCATIONAL DEMO OUTPUT — NOT A MEDICAL DIAGNOSIS")
        print("Predicted illustrative category:", predicted_category)
        print("Classifier output probabilities:")
        for category in ["Low", "Moderate", "High"]:
            print(f"  {category}: {probability_map.get(category, 0):.1%}")

        prob_df = pd.DataFrame({
            "Category": list(probability_map.keys()),
            "Model probability": list(probability_map.values())
        })
        ax = sns.barplot(
            data=prob_df, x="Category", y="Model probability",
            order=["Low", "Moderate", "High"]
        )
        ax.set_title("Classifier Output Probabilities (not disease probabilities)")
        ax.set_ylim(0, 1)
        plt.show()

analyze_button.on_click(run_prediction)

display(widgets.VBox([
    age_input, bmi_input, sys_input, dia_input,
    glucose_input, hr_input, chol_input, activity_input,
    analyze_button, result_output
]))


# Cell 10 — Download a CSV report
# This creates a report from one fictional/demo assessment.

from google.colab import files
from datetime import datetime

demo_input = {
    "age": age_input.value,
    "bmi": bmi_input.value,
    "systolic_bp": sys_input.value,
    "diastolic_bp": dia_input.value,
    "glucose": glucose_input.value,
    "heart_rate": hr_input.value,
    "cholesterol": chol_input.value,
    "activity_minutes": activity_input.value,
}

sample = pd.DataFrame([demo_input], columns=FEATURES)
predicted_category = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]
probability_map = dict(zip(model.classes_, probabilities))

report = pd.DataFrame([{
    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    **demo_input,
    "demo_predicted_category": predicted_category,
    "model_confidence_not_clinical": float(max(probabilities)),
    "disclaimer": "Synthetic educational demo only; not a medical diagnosis."
}])

report_path = "health_risk_demo_report.csv"
report.to_csv(report_path, index=False)
display(report)
files.download(report_path)


# ## Project conclusion
# 
# This notebook demonstrates a complete introductory-to-advanced machine-learning workflow:
# 1. Synthetic data generation
# 2. Data cleaning checks and exploratory analysis
# 3. Train/test split
# 4. Random Forest model training
# 5. Holdout evaluation
# 6. Feature importance
# 7. Interactive prediction demo
# 8. CSV report export
# 
# **For your portfolio:** explain that the dataset and target labels are synthetic. Do not describe the model as medically validated. A production medical system would require suitable clinical datasets, expert review, privacy controls, rigorous validation, and regulatory review.

