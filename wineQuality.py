import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle

st.title("🍷 Wine Quality Prediction - ML Models Comparison")

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Lab 22 winequality-red.csv")

wine_df = load_data()

st.subheader("Dataset Preview")
st.dataframe(wine_df.head())

# -------------------------------
# Visualization
# -------------------------------
st.subheader("Wine Quality Distribution")

fig1 = plt.figure(figsize=(8,5))
sns.countplot(x='quality', data=wine_df)
st.pyplot(fig1)

# -------------------------------
# Prepare Data
# -------------------------------
X = wine_df.drop(columns=['quality'])
y = wine_df['quality']

oversample = SMOTE(k_neighbors=4)
X, y = oversample.fit_resample(X.fillna(0), y)

# -------------------------------
# Training Function
# -------------------------------
def classify(model):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model.fit(X_train, y_train)
    return model.score(X_test, y_test) * 100

# -------------------------------
# Train Models
# -------------------------------
linear_acc = classify(LogisticRegression(max_iter=2000))
tree_acc = classify(DecisionTreeClassifier())
rf_acc = classify(RandomForestClassifier())
svm_acc = classify(SVC())

# -------------------------------
# Results
# -------------------------------
st.subheader("Model Accuracy (%)")

accuracy = [linear_acc, tree_acc, rf_acc, svm_acc]
models = ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]

fig2 = plt.figure(figsize=(8,5))
sns.barplot(x=accuracy, y=models)
st.pyplot(fig2)

# -------------------------------
# Save Random Forest Model
# -------------------------------
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

with open("finalRF_model.sav", "wb") as file:
    pickle.dump(rf_model, file)

st.success("Random Forest model trained and saved successfully!")

# -------------------------------
# Prediction Section
# -------------------------------
st.subheader("Predict Wine Quality")

input_data = []
for col in X.columns:
    val = st.number_input(col)
    input_data.append(val)

if st.button("Predict"):
    result = rf_model.predict([input_data])
    st.success(f"Predicted Wine Quality: {result[0]}")
