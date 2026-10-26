🏥 Disease Detection Based on Symptoms

A Machine Learning–based web application that predicts possible diseases based on user-selected symptoms.
The system also provides disease meaning, severity level, recommended specialist, and self-care precautions.

📌 Project Overview

This project uses a trained Machine Learning model to analyze symptoms entered by a user and predict the most likely disease.
It is built using Python, Streamlit, and Scikit-learn, with additional medical data stored in Excel files.

⚠️ Note: This application is for educational purposes only and not a replacement for professional medical diagnosis.


🚀 Features

✅ Symptom-based disease prediction

✅ Interactive web interface using Streamlit

✅ Severity level detection (Low / Moderate / High)

✅ Disease explanation and meaning

✅ Suggested medical specialist

✅ Self-care and precaution guidelines

✅ Health helpline access

🧠 Machine Learning Details

Algorithm Used: Decision Tree Classifier

Input: Binary symptom vector (1 = present, 0 = absent)

Output: Predicted disease name

Dataset:

Training.csv – used for model training

Testing.csv – used for validation

The trained model is stored as:

model copy.pkl

🗂 Project Structure
disease-detection-based-on-symptoms/
│
├── app.py                    # Streamlit application
├── train_model.py            # Model training script
├── model copy.pkl            # Trained ML model
├── Training.csv              # Training dataset
├── Testing.csv               # Testing dataset
├── Precautions.xlsx          # Disease precautions & specialist info
├── Severity.xlsx             # Symptom severity levels
├── requirements.txt          # Required libraries
├── assets/                   # Images used in UI
└── README.md                 # Project documentation

🖥️ How the Application Works

User selects multiple symptoms

User selects severity level

Symptoms are converted into a binary feature vector

The ML model predicts the disease

Additional details are fetched from Excel files:

Disease meaning

Specialist recommendation

Self-care instructions

Severity level

Output is displayed on the web interface

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/Ram-dwarampudi/disease-detection-based-on-symptoms.git
cd disease-detection-based-on-symptoms

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Train the Model (Optional)
python train_model.py

4️⃣ Run the Application
streamlit run app.py

📦 Requirements

Python 3.8+

Streamlit

NumPy

Pandas

Scikit-learn

OpenPyXL

(See requirements.txt for full list)

⚠️ Disclaimer

This system provides predictions based on symptoms and is not a medical diagnosis tool.
Always consult a qualified healthcare professional for medical advice.

👨‍💻 Author

Ram (Dwarampudi Venkata Sai Rama Gopala Reddy)

B.Tech – Computer Science & Business Systems

Vishnu Institute of Technology, Bhimavaram

GitHub: https://github.com/Ram-dwarampudi

🌟 Future Enhancements

🔹 Add probability score for predictions

🔹 Improve accuracy using ensemble models

🔹 Add login & user history

🔹 Deploy on Streamlit Cloud

🔹 Add visual analytics
