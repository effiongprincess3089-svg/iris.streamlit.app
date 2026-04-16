
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Page Configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌸 Iris Flower Classification App")
st.markdown("""
This app demonstrates **multiple machine learning models** for classifying Iris flowers 
based on sepal and petal measurements.
""")

# Load Data
@st.cache_data
def load_iris_data():
    iris = load_iris()
    return iris.data, iris.target, iris.feature_names, iris.target_names

X, y, feature_names, target_names = load_iris_data()

# Train Models
@st.cache_resource
def train_models():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=300, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM (Linear)": SVC(kernel='linear', probability=True, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    performance = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        performance[name] = {
            "model": model,
            "accuracy": acc,
            "cm": cm,
            "report": classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        }
    
    return models, performance

models_dict, performance = train_models()

# Sidebar
st.sidebar.header("🌺 Flower Measurements")

sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
sw = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.3, 0.1)
pw = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.3, 0.1)

st.sidebar.markdown("---")
selected_model_name = st.sidebar.selectbox(
    "Choose Classification Model",
    options=list(models_dict.keys()),
    index=3  # KNN as default
)

# Main Prediction
input_data = np.array([[sl, sw, pl, pw]])
model = performance[selected_model_name]["model"]
pred = model.predict(input_data)[0]
probs = model.predict_proba(input_data)[0]

predicted_species = target_names[pred].title()

st.subheader("🎯 Prediction")
st.success(f"**{predicted_species}**")
st.metric("Confidence", f"{max(probs)*100:.1f}%")

# Probability Chart
st.subheader("📊 Prediction Probabilities")
prob_df = pd.DataFrame({
    "Species": [name.title() for name in target_names],
    "Probability": probs
})
st.bar_chart(prob_df.set_index("Species"), color="#4CAF50")

# Performance Section
st.markdown("---")
st.subheader("📈 Model Performance")

perf_data = [{"Model": name, "Accuracy (%)": f"{p['accuracy']*100:.2f}%"} 
             for name, p in performance.items()]
st.dataframe(pd.DataFrame(perf_data).sort_values("Accuracy (%)", ascending=False), 
             use_container_width=True)

best_model = max(performance, key=lambda k: performance[k]["accuracy"])
st.success(f"🏆 Best Model: **{best_model}** ({performance[best_model]['accuracy']*100:.2f}%)")

# Detailed View
with st.expander(f"📋 Confusion Matrix - {selected_model_name}", expanded=False):
    cm_df = pd.DataFrame(
        performance[selected_model_name]["cm"],
        index=[f"Actual {t.title()}" for t in target_names],
        columns=[f"Predicted {t.title()}" for t in target_names]
    )
    st.dataframe(cm_df)
