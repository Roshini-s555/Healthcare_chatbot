# 🩺 Advanced AI Healthcare Assistant & Analytics Dashboard

An intelligent, full-stack healthcare chatbot and analytics web application powered by **Machine Learning (TF-IDF & Logistic Regression)** trained on **25,000+ medical records** from Kaggle. Built with Python and Streamlit, this application features a real-time conversational interface, automated dataset ingestion, and a patient vitals analytics dashboard.

---

## 🚀 Live Demo
Access the live application here: [Streamlit Cloud Deployment](https://healthcarechatbot-modmsxwek9uecurdqnvkto.streamlit.app/)

---

## ✨ Key Features

* **🤖 Machine Learning Classification:** Replaces basic keyword lookups with natural language processing (`scikit-learn` TF-IDF Vectorization and Logistic Regression) to analyze unstructured symptom descriptions and predict matching medical conditions.
* **💬 Conversational UI:** Interactive multi-turn chat interface leveraging `st.session_state` and Streamlit chat elements for a seamless user experience.
* **📊 Vitals & BMI Analytics Dashboard:** Dynamic sidebar tracking patient vitals (Age, Gender, Height, Weight), automated BMI calculations, health risk categorization, and historical trend line charts.
* **🔄 Automated Dataset Ingestion:** Utilizes `kagglehub` to securely fetch and cache the 25,000-record medical dataset directly on startup.

---

## 🛠️ Tech Stack

* **Frontend & UI:** Streamlit
* **Machine Learning & NLP:** Scikit-learn (`TfidfVectorizer`, `LogisticRegression`)
* **Data Processing:** Pandas, NumPy
* **Data Source:** Kaggle (`healthcare-symptomsdisease-classification-dataset`)
* **Version Control:** Git & GitHub
* Create and Activate a Virtual Environment:

## ⚙️ Installation & Local Setup

Follow these steps to run the application locally on your machine:

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Roshini-s555/Healthcare_chatbot.git](https://github.com/Roshini-s555/Healthcare_chatbot.git)
   cd Healthcare_chatbot

   Bash
python -m venv venv
# On Windows PowerShell:
venv\Scripts\Activate.ps1
Install Dependencies:

Bash
pip install -r requirements.txt
Run the Streamlit App:

Bash
streamlit run app.py
📂 Project Structure
Plaintext
healthcare_chatbot/
│
├── app.py              # Main Streamlit application & ML model pipeline
├── requirements.txt    # Project dependencies and package versions
└── README.md           # Project documentation
⚠️ Medical Disclaimer
This tool is designed for educational, research, and preliminary screening purposes only. It does not replace professional medical diagnosis, advice, or treatment.


---

<ElicitationsGroup>
  <Elicitation>Were you able to paste the README content and commit the file successfully on GitHub?</Elicitation>
</ElicitationsGroup>

---
