import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# -----------------------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MediScan AI | Symptom-Based Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    /* ---- HARD CONTRAST LOCK -------------------------------------------
       Streamlit switches text/background colors based on the viewer's
       system theme (light/dark). Combined with .streamlit/config.toml
       (which pins the app to light theme), these rules force every piece
       of text — including dropdown menus rendered outside the main app
       container — to stay dark-on-light no matter what. -------------- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f4f7fb 0%, #eef2f9 100%) !important;
    }
    [data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #eef1f6;
    }
    .stApp, .stApp p, .stApp span, .stApp li, .stApp label,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    h1, h2, h3, h4, h5, h6,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #0f172a !important;
    }
    /* Dropdown menus (multiselect options) render in a portal outside
       .stApp, so they need their own explicit override */
    div[data-baseweb="popover"], div[data-baseweb="menu"] {
        background: #ffffff !important;
    }
    div[data-baseweb="popover"] *, div[data-baseweb="menu"] *, li[role="option"] {
        color: #0f172a !important;
    }
    [data-baseweb="tag"] { color: #ffffff !important; }
    [data-baseweb="tag"] span { color: #ffffff !important; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 50%, #06b6d4 100%);
        border-radius: 20px;
        padding: 2.6rem 2.4rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 30px rgba(15, 118, 110, 0.25);
        color: white;
    }
    .hero h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 2.3rem;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .hero p {
        font-size: 1.05rem;
        opacity: 0.94;
        margin: 0;
        max-width: 640px;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        border: 1px solid rgba(255,255,255,0.35);
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        margin-bottom: 0.9rem;
        backdrop-filter: blur(4px);
    }

    /* Form card */
    [data-testid="stForm"], .card {
        background: white !important;
        border-radius: 18px !important;
        padding: 1.8rem 1.8rem !important;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.06) !important;
        border: 1px solid #eef1f6 !important;
        margin-bottom: 1.4rem !important;
    }
    [data-testid="stForm"] h3, .card h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        color: #0f172a;
        margin-top: 0;
    }

    /* Step badges */
    .step-label {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 26px; height: 26px;
        background: #0f766e;
        color: white;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 8px;
    }

    /* Predict button */
    div.stButton > button, button[kind="formSubmit"] {
        background: linear-gradient(135deg, #0f766e, #06b6d4);
        color: white;
        font-weight: 700;
        font-size: 1.02rem;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.4rem;
        width: 100%;
        box-shadow: 0 6px 16px rgba(15, 118, 110, 0.3);
        transition: transform 0.15s ease;
    }
    div.stButton > button:hover, button[kind="formSubmit"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 22px rgba(15, 118, 110, 0.38);
    }

    /* Result cards */
    .result-header {
        background: linear-gradient(135deg, #fee2e2, #fef3c7);
        border-left: 6px solid #dc2626;
        border-radius: 14px;
        padding: 1.3rem 1.6rem;
        margin-bottom: 1.1rem;
    }
    .result-header .label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #991b1b;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .result-header .disease-name {
        font-family: 'Poppins', sans-serif;
        font-size: 1.7rem;
        font-weight: 800;
        color: #7f1d1d;
        margin: 0;
    }

    .info-tile {
        background: #f8fafc;
        border-radius: 14px;
        border: 1px solid #eef1f6;
        padding: 1.1rem 1.3rem;
        height: 100%;
    }
    .info-tile .icon { font-size: 1.4rem; }
    .info-tile .title {
        font-weight: 700;
        color: #0f172a;
        font-size: 0.92rem;
        margin: 0.35rem 0 0.3rem 0;
        text-transform: uppercase;
        letter-spacing: 0.4px;
    }
    .info-tile .body {
        color: #334155;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .severity-pill {
        display: inline-block;
        padding: 5px 16px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.3px;
    }
    .sev-low { background: #dcfce7; color: #166534; }
    .sev-moderate { background: #fef9c3; color: #854d0e; }
    .sev-high { background: #fee2e2; color: #991b1b; }

    .sidebar-card {
        background: white;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #eef1f6;
        margin-bottom: 1rem;
    }

    .helpline {
        background: linear-gradient(135deg, #0f766e, #06b6d4);
        color: white;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        text-align: center;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# LOAD MODEL & DATA
# ----------------------------------------------------------------------------
model_file = 'model copy.pkl' if os.path.exists('model copy.pkl') and os.path.getsize('model copy.pkl') > 100 else 'model.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

disease_list = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing',
    'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
    'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_urination', 'fatigue',
    'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
    'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite',
    'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea',
    'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
    'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise',
    'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes',
    'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
    'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
    'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity',
    'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
    'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts',
    'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness',
    'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
    'loss_of_smell', 'bladder_discomfort', 'foul_smell_of_urine', 'continuous_feel_of_urine',
    'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression',
    'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
    'abnormal_menstruation', 'dischromic_patches', 'watering_from_eyes', 'increased_appetite',
    'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
    'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections',
    'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption',
    'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations',
    'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
    'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
    'red_sore_around_nose', 'yellow_crust_ooze']

severity_options = ['Low', 'Moderate', 'High']

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:

    st.markdown("""
    <div class="sidebar-card">
        <b>🩺 About MediScan AI</b><br><br>
        <span style="color:#475569; font-size:0.92rem; line-height:1.55;">
        A machine-learning powered symptom checker that suggests a likely
        condition, its severity, a recommended specialist, and self-care tips
        — built with a Decision Tree Classifier.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <b>📋 How it works</b><br><br>
        <span style="color:#475569; font-size:0.9rem; line-height:1.7;">
        <span class="step-label">1</span> Select your symptoms<br><br>
        <span class="step-label">2</span> Choose severity level<br><br>
        <span class="step-label">3</span> Get an instant prediction
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="helpline">
        📞 Health Helpline<br>
        <span style="font-size:1.3rem;">104</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Built by Ram · B.Tech CSBS, Vishnu Institute of Technology")

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="badge">⚡ AI-POWERED SYMPTOM CHECKER</div>
    <h1>MediScan AI — Symptom-Based Disease Prediction</h1>
    <p>Select your symptoms below and get an instant, data-driven prediction along with
    severity insight, specialist recommendations, and self-care guidance.</p>
</div>
""", unsafe_allow_html=True)

st.warning("⚠️ This tool is for educational purposes only and is **not** a substitute for professional medical diagnosis. Always consult a qualified doctor.", icon="⚠️")

# ----------------------------------------------------------------------------
# INPUT FORM
# ----------------------------------------------------------------------------
with st.form("my_form"):
    st.markdown("### 🧾 Tell us how you're feeling")
    col1, col2 = st.columns([2.2, 1])

    with col1:
        st.markdown("**<span class='step-label'>1</span> Select your symptoms**", unsafe_allow_html=True)
        Symptoms = st.multiselect(
            " ",
            disease_list,
            placeholder="Start typing to search symptoms...",
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("**<span class='step-label'>2</span> Severity level**", unsafe_allow_html=True)
        Severity = st.radio(" ", severity_options, label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict My Condition")

# Fallback disease information for conditions not present in Precautions.xlsx
FALLBACK_DISEASE_INFO = {
    "Fungal infection": {
        "meaning": "A fungal infection (mycosis) is an inflammatory skin condition caused by fungi, commonly leading to rashes, itching, and skin eruptions (e.g. ringworm, athlete's foot).",
        "specialist": "Dermatologist",
        "self_care": "Keep the affected skin clean and dry, wear loose-fitting breathable clothing, avoid sharing personal items, and apply prescribed antifungal medications."
    },
    "Allergy": {
        "meaning": "An exaggerated immune response to foreign substances (allergens) such as pollen, pet dander, foods, or dust, causing sneezing, rashes, or watery eyes.",
        "specialist": "Allergist / Immunologist",
        "self_care": "Identify and avoid known allergens, stay indoors during peak pollen counts, use saline nasal rinses, and take antihistamines if recommended by a doctor."
    },
    "Acne": {
        "meaning": "A common skin condition that occurs when hair follicles become plugged with sebum and dead skin cells, causing pimples, blackheads, or cysts.",
        "specialist": "Dermatologist",
        "self_care": "Cleanse gently twice daily, avoid scrubbing or picking at blemishes, use oil-free non-comedogenic skincare products, and maintain hydration."
    },
    "Peptic ulcer diseae": {
        "meaning": "Painful sores in the lining of the stomach or small intestine, often caused by Helicobacter pylori bacterial infection or frequent NSAID painkiller use.",
        "specialist": "Gastroenterologist",
        "self_care": "Eat small and regular meals, limit acidic and spicy foods, avoid tobacco and alcohol, and avoid self-medicating with ibuprofen or aspirin."
    },
    "(vertigo) Paroymsal  Positional Vertigo": {
        "meaning": "Benign Paroxysmal Positional Vertigo (BPPV) is a sensory disorder of the inner ear causing brief episodes of dizziness or spinning sensations with head movement.",
        "specialist": "ENT Specialist / Neurologist",
        "self_care": "Move head slowly, sit down immediately upon feeling dizzy, use good lighting if walking at night, and consult an ENT specialist for canalith repositioning."
    },
    "Hyperthyroidism": {
        "meaning": "An overactive thyroid condition where the thyroid gland produces excess thyroid hormones, accelerating the body's metabolic rate.",
        "specialist": "Endocrinologist",
        "self_care": "Maintain balanced caloric and nutritional intake, minimize caffeine and stress, avoid excess dietary iodine, and schedule regular thyroid checks."
    },
    "hepatitis A": {
        "meaning": "A viral liver infection typically transmitted through contaminated food or water, leading to fatigue, nausea, abdominal discomfort, and jaundice.",
        "specialist": "Hepatologist / Gastroenterologist",
        "self_care": "Get plenty of rest, maintain hydration with electrolytes, eat bland and nutritious foods, practice meticulous hand hygiene, and avoid alcohol."
    }
}

# ----------------------------------------------------------------------------
# PREDICTION LOGIC
# ----------------------------------------------------------------------------
if submitted:
    if len(Symptoms) != 0:
        df = pd.read_excel('Precautions.xlsx')
        df1 = pd.read_excel('Severity.xlsx')
        df1["symptoms"] = df1["symptoms"].str.replace(' ', '_')

        data, Severity1 = [], []
        for j in Symptoms:
            z = df1[df1['symptoms'].str.contains(j, case=False, regex=False)]
            if len(z) > 0 and 'Severity Level' in z.columns:
                Severity1.append(z['Severity Level'].iloc[0])

        for j in disease_list:
            data.append(1 if j in Symptoms else 0)

        feature_names = getattr(model, 'feature_names_in_', None)
        if feature_names is not None and len(feature_names) == len(data):
            features_df = pd.DataFrame([data], columns=feature_names)
            output = model.predict(features_df)
        else:
            output = model.predict(np.array(data).reshape(1, -1))
        pred_disease = output[0]

        # Case-insensitive & trimmed matching in Precautions.xlsx
        matched_df = df[df['Diseases'].astype(str).str.strip().str.lower() == pred_disease.strip().lower()]
        if matched_df.empty:
            matched_df = df[df['Diseases'].astype(str).apply(
                lambda x: x.strip().lower() in pred_disease.lower() or pred_disease.lower() in x.strip().lower()
            )]

        if not matched_df.empty:
            meaning = matched_df["Disease Meaning"].values[0]
            specialist = matched_df["Specalist"].values[0]
            self_care = matched_df["SELF CARE"].values[0]
        elif pred_disease in FALLBACK_DISEASE_INFO:
            fallback = FALLBACK_DISEASE_INFO[pred_disease]
            meaning = fallback["meaning"]
            specialist = fallback["specialist"]
            self_care = fallback["self_care"]
        else:
            meaning = "A health condition characterized by the symptoms reported."
            specialist = "General Physician"
            self_care = "Rest well, drink plenty of fluids, monitor your symptoms, and visit a doctor for clinical testing."

        final_severity = Severity if Severity == 'High' else (max(Severity1, key=Severity1.count) if Severity1 else Severity)
        sev_class = {"Low": "sev-low", "Moderate": "sev-moderate", "High": "sev-high"}.get(final_severity, "sev-moderate")

        st.markdown(f"""
        <div class="result-header">
            <div class="label">Predicted Condition</div>
            <p class="disease-name">🩹 {pred_disease}</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="info-tile">
                <div class="icon">📖</div>
                <div class="title">What it means</div>
                <div class="body">{meaning}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="info-tile">
                <div class="icon">👨‍⚕️</div>
                <div class="title">Consult a</div>
                <div class="body">{specialist}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="info-tile">
                <div class="icon">📊</div>
                <div class="title">Severity level</div>
                <div class="body"><span class="severity-pill {sev_class}">{final_severity}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="info-tile">
            <div class="icon">🌿</div>
            <div class="title">Self-care & precautions</div>
            <div class="body">{self_care}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 Remember: this prediction is a guide, not a diagnosis. If symptoms persist or worsen, please see a doctor.")
    else:
        st.warning("👆 Please select at least one symptom to get a prediction.")
