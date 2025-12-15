import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------
# Page config
# ----------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction – Model Comparison App")

st.write("""
This app trains **four machine learning models** to predict customer churn
and compares their performance.
""")

# ----------------------------------
# File upload
# ----------------------------------
uploaded_file = st.file_uploader("Upload Cleaned Customer Churn CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # ----------------------------------
    # Split features and target
    # ----------------------------------
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    test_size = st.slider("Test size", 0.1, 0.4, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.success("Data successfully split into train & test sets")

    # ----------------------------------
    # Train models
    # ----------------------------------
    if st.button("🚀 Train Models"):
        results = []

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42
            )
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cr = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Confusion Matrix": cm,
                "Classification Report": cr
            })

        # ----------------------------------
        # Display results
        # ----------------------------------
        st.subheader("📈 Model Performance")

        for res in results:
            st.markdown(f"## 🔹 {res['Model']}")
            st.write(f"**Accuracy:** {res['Accuracy']:.4f}")

            # Classification report
            st.write("### Classification Report")
            st.dataframe(pd.DataFrame(res["Classification Report"]).transpose())

            # Confusion matrix plot
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots()
            ax.matshow(res["Confusion Matrix"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        # ----------------------------------
        # Model comparison table
        # ----------------------------------
        st.subheader("🏆 Model Comparison Summary")

        comparison_df = pd.DataFrame({
            "Model": [r["Model"] for r in results],
            "Accuracy": [r["Accuracy"] for r in results]
        }).sort_values(by="Accuracy", ascending=False)

        st.dataframe(comparison_df)

else:
    st.info("👆 Please upload the **Cleaned_Customer_Churn.csv** file to begin.")
