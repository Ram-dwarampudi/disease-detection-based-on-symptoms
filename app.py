import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('model copy.pkl', 'rb'))

disease_list = ['itching',
'skin_rash',
'nodal_skin_eruptions',
'continuous_sneezing',
'shivering',
'chills',
'joint_pain',
'stomach_pain',
'acidity',
'ulcers_on_tongue',
'muscle_wasting',
'vomiting',
'burning_micturition',
'spotting_urination',
'fatigue',
'weight_gain',
'anxiety',
'cold_hands_and_feets',
'mood_swings',
'weight_loss',
'restlessness',
'lethargy',
'patches_in_throat',
'irregular_sugar_level',
'cough',
'high_fever',
'sunken_eyes',
'breathlessness',
'sweating',
'dehydration',
'indigestion',
'headache',
'yellowish_skin',
'dark_urine',
'nausea',
'loss_of_appetite',
'pain_behind_the_eyes',
'back_pain',
'constipation',
'abdominal_pain',
'diarrhoea',
'mild_fever',
'yellow_urine',
'yellowing_of_eyes',
'acute_liver_failure',
'fluid_overload',
'swelling_of_stomach',
'swelled_lymph_nodes',
'malaise',
'blurred_and_distorted_vision',
'phlegm',
'throat_irritation',
'redness_of_eyes',
'sinus_pressure',
'runny_nose',
'congestion',
'chest_pain',
'weakness_in_limbs',
'fast_heart_rate',
'pain_during_bowel_movements',
'pain_in_anal_region',
'bloody_stool',
'irritation_in_anus',
'neck_pain',
'dizziness',
'cramps',
'bruising',
'obesity',
'swollen_legs',
'swollen_blood_vessels',
'puffy_face_and_eyes',
'enlarged_thyroid',
'brittle_nails',
'swollen_extremeties',
'excessive_hunger',
'extra_marital_contacts',
'drying_and_tingling_lips',
'slurred_speech',
'knee_pain',
'hip_joint_pain',
'muscle_weakness',
'stiff_neck',
'swelling_joints',
'movement_stiffness',
'spinning_movements',
'loss_of_balance',
'unsteadiness',
'weakness_of_one_body_side',
'loss_of_smell',
'bladder_discomfort',
'foul_smell_of_urine',
'continuous_feel_of_urine',
'passage_of_gases',
'internal_itching',
'toxic_look_(typhos)',
'depression',
'irritability',
'muscle_pain',
'altered_sensorium',
'red_spots_over_body',
'belly_pain',
'abnormal_menstruation',
'dischromic_patches',
'watering_from_eyes',
'increased_appetite',
'polyuria',
'family_history',
'mucoid_sputum',
'rusty_sputum',
'lack_of_concentration',
'visual_disturbances',
'receiving_blood_transfusion',
'receiving_unsterile_injections',
'coma',
'stomach_bleeding',
'distention_of_abdomen',
'history_of_alcohol_consumption',
'fluid_overload',
'blood_in_sputum',
'prominent_veins_on_calf',
'palpitations',
'painful_walking',
'pus_filled_pimples',
'blackheads',
'scurring',
'skin_peeling',
'silver_like_dusting',
'small_dents_in_nails',
'inflammatory_nails',
'blister',
'red_sore_around_nose',
'yellow_crust_ooze']

st.title("Symptom-Based Disease Prediction")
st.info("⚠️ This application is for educational purposes only and not a medical diagnosis. Please consult a qualified doctor.")

list = ['Low','Moderate','High']

st.image('WhatsApp Image 2025-05-29 at 21.38.32_5496b124.jpg')


with st.form("my_form"):
    Symptoms = st.multiselect('select your Symptoms:',disease_list)
    Severity = st.radio('select your Severity:',list)
    submitted = st.form_submit_button("Predict")

if submitted:
   if len(Symptoms) != 0:
    df = pd.read_excel('Precautions.xlsx')
    df1 = pd.read_excel('Severity.xlsx')
    df1["symptoms"]=df1["symptoms"].str.replace(' ', '_')
    data,Severity1 = [],[]
    for j in Symptoms:
        z = df1[df1['symptoms'].str.contains(j)]
        Severity1.append(z['Severity Level'].iloc[0])
    for j in disease_list:
     if j in Symptoms:
      data.append(1)
     else:
      data.append(0)
    output = model.predict(np.array(data).reshape(1, -1))
    st.write("you might have this disease ")
    df = df[df['Diseases'].str.contains(output[0])]
    st.error(output[0])
    try :
        st.write(df["Disease Meaning"].values[0])
        st.success( "You Can Consultant : " + df["Specalist"].values[0])
        st.write("Thing To Do:")
        st.warning(df["SELF CARE"].values[0])
        st.write("Severity :")
        if Severity=='High':
            st.error(Severity)
        else:
            st.error(max(Severity1, key=Severity1.count))

    except:
        print("New Disease")


   else:
    st.warning("please enter your Symptoms")

