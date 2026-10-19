import streamlit as st
import pandas as pd
import kagglehub
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(page_title="Advanced Healthcare AI Chatbot", page_icon="🤖", layout="wide")

st.title("AI Healthcare Assistant & Analytics Dashboard 🩺")
st.markdown("Powered by Machine Learning (TF-IDF & Logistic Regression) on 25,000 Kaggle Medical Records.")

# 1. Load Dataset and Train ML Model
@st.cache_resource
def load_and_train_model():
    path = kagglehub.dataset_download("kundanbedmutha/healthcare-symptomsdisease-classification-dataset")
    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                csv_file = os.path.join(root, file)
                break
    
    df = pd.read_csv(csv_file)
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Symptoms'])
    y = df['Disease']
    
    model = LogisticRegression(max_iter=300)
    model.fit(X, y)
    return model, vectorizer

with st.spinner("Initializing AI model and medical knowledge base..."):
    model, vectorizer = load_and_train_model()

# 2. Sidebar for Vitals & Trend Analytics Dashboard
st.sidebar.header("👤 Patient Vitals & Profile")
age = st.sidebar.slider("Age", 1, 90, 25)
gender = st.sidebar.selectbox("Gender", ("Male", "Female", "Other"))
height = st.sidebar.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)
weight = st.sidebar.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=70.0)

# Calculate current BMI
bmi = round(weight / ((height / 100) ** 2), 1)
bmi_status = "Underweight" if bmi < 18.5 else "Healthy" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"

st.sidebar.metric(label="Calculated BMI", value=bmi, delta=bmi_status)

# Initialize Session History for Health Tracking Dashboard
if "bmi_history" not in st.session_state:
    st.session_state.bmi_history = []

if st.sidebar.button("Log Vitals Snapshot"):
    st.session_state.bmi_history.append({"Record": len(st.session_state.bmi_history) + 1, "BMI": bmi})
    st.sidebar.success("Vitals logged successfully!")

# Display Trend Chart in Sidebar if logs exist
if st.session_state.bmi_history:
    st.sidebar.subheader("📈 BMI Session Trends")
    df_history = pd.DataFrame(st.session_state.bmi_history).set_index("Record")
    st.sidebar.line_chart(df_history)

# 3. Initialize Chat History in Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! I am your AI healthcare assistant. Based on your sidebar profile (Age: {age}, BMI: {bmi}), describe your symptoms and I will predict the potential condition."}
    ]

# Main Chat Interface Layout
st.subheader("💬 Clinical Consultation Chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Chat Input
if prompt := st.chat_input("Type your symptoms here (e.g., fever, cough, fatigue)..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running machine learning classification..."):
            user_vector = vectorizer.transform([prompt.lower()])
            predicted_disease = model.predict(user_vector)[0]
            
            response = f"🔍 **AI Diagnostic Prediction:** Based on your symptom patterns, the most matching condition is **{predicted_disease}**.\n\n"
            if age > 50:
                response += "⚠️ *Clinical Advisory: Given your age bracket (>50), please monitor closely and consult a specialist if symptoms persist.*\n\n"
            response += "*Disclaimer: This machine learning tool is for academic screening purposes only.*"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})