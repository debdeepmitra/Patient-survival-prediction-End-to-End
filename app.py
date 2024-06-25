import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from helper import encode, get_prediction

model = tf.keras.models.load_model(r'Models/model.h5')

st.set_page_config(page_title ='Patient Survival Prediction', page_icon = '🩺', layout ='wide')
st.markdown("<h1 style = 'text-align: center;'>Patient Survival Prediction 🩺</h1>", unsafe_allow_html=True)

# Instructions
st.write("Instructions: Download the sample input file using the following button, fill it up properly, and upload it to see the result!")


def file_download_link(filepath):
    with open(filepath, 'rb') as file:
        st.download_button(
            label="Download Sample File",
            data=file,
            file_name=filepath.split('/')[-1],  # Extract file name from the path
            mime="application/vnd.ms-excel"
        )


file_download_link('sample_input_file.xlsx')

uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file is None:
    st.write("No file uploaded yet.")
else:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)
    
    # Display the dataframe
    st.write("Here's a preview of your file:")
    st.dataframe(df)

    df_transpose = df.set_index('Parameters').T   
    data = df_transpose[['encounter_id', 'patient_id', 'hospital_id', 'hospital_death', 'age', 'bmi', 'elective_surgery', 'ethnicity', 'gender', 'height', 'hospital_admit_source', 'icu_admit_source', 'icu_id', 'icu_stay_type', 'icu_type', 'pre_icu_los_days', 'readmission_status', 'weight', 'albumin_apache', 'apache_2_diagnosis', 'apache_3j_diagnosis', 'apache_post_operative', 'arf_apache', 'bilirubin_apache', 'bun_apache', 'creatinine_apache', 'fio2_apache', 'gcs_eyes_apache', 'gcs_motor_apache', 'gcs_unable_apache', 'gcs_verbal_apache', 'glucose_apache', 'heart_rate_apache', 'hematocrit_apache', 'intubated_apache', 'map_apache', 'paco2_apache', 'paco2_for_ph_apache', 'pao2_apache', 'ph_apache', 'resprate_apache', 'sodium_apache', 'temp_apache', 'urineoutput_apache', 'ventilated_apache', 'wbc_apache', 'd1_diasbp_invasive_max', 'd1_diasbp_invasive_min', 'd1_diasbp_max', 'd1_diasbp_min', 'd1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min', 'd1_heartrate_max', 'd1_heartrate_min', 'd1_mbp_invasive_max', 'd1_mbp_invasive_min', 'd1_mbp_max', 'd1_mbp_min', 'd1_mbp_noninvasive_max', 'd1_mbp_noninvasive_min', 'd1_resprate_max', 'd1_resprate_min', 'd1_spo2_max', 'd1_spo2_min', 'd1_sysbp_invasive_max', 'd1_sysbp_invasive_min', 'd1_sysbp_max', 'd1_sysbp_min', 'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 'd1_temp_max', 'd1_temp_min', 'h1_diasbp_invasive_max', 'h1_diasbp_invasive_min', 'h1_diasbp_max', 'h1_diasbp_min', 'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 'h1_heartrate_max', 'h1_heartrate_min', 'h1_mbp_invasive_max', 'h1_mbp_invasive_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min', 'h1_resprate_max', 'h1_resprate_min', 'h1_spo2_max', 'h1_spo2_min', 'h1_sysbp_invasive_max', 'h1_sysbp_invasive_min', 'h1_sysbp_max', 'h1_sysbp_min', 'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min', 'h1_temp_max', 'h1_temp_min', 'd1_albumin_max', 'd1_albumin_min', 'd1_bilirubin_max', 'd1_bilirubin_min', 'd1_bun_max', 'd1_bun_min', 'd1_calcium_max', 'd1_calcium_min', 'd1_creatinine_max', 'd1_creatinine_min', 'd1_glucose_max', 'd1_glucose_min', 'd1_hco3_max', 'd1_hco3_min', 'd1_hemaglobin_max', 'd1_hemaglobin_min', 'd1_hematocrit_max', 'd1_hematocrit_min', 'd1_inr_max', 'd1_inr_min', 'd1_lactate_max', 'd1_lactate_min', 'd1_platelets_max', 'd1_platelets_min', 'd1_potassium_max', 'd1_potassium_min', 'd1_sodium_max', 'd1_sodium_min', 'd1_wbc_max', 'd1_wbc_min', 'h1_albumin_max', 'h1_albumin_min', 'h1_bilirubin_max', 'h1_bilirubin_min', 'h1_bun_max', 'h1_bun_min', 'h1_calcium_max', 'h1_calcium_min', 'h1_creatinine_max', 'h1_creatinine_min', 'h1_glucose_max', 'h1_glucose_min', 'h1_hco3_max', 'h1_hco3_min', 'h1_hemaglobin_max', 'h1_hemaglobin_min', 'h1_hematocrit_max', 'h1_hematocrit_min', 'h1_inr_max', 'h1_inr_min', 'h1_lactate_max', 'h1_lactate_min', 'h1_platelets_max', 'h1_platelets_min', 'h1_potassium_max', 'h1_potassium_min', 'h1_sodium_max', 'h1_sodium_min', 'h1_wbc_max', 'h1_wbc_min', 'd1_arterial_pco2_max', 'd1_arterial_pco2_min', 'd1_arterial_ph_max', 'd1_arterial_ph_min', 'd1_arterial_po2_max', 'd1_arterial_po2_min', 'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min', 'h1_arterial_pco2_max', 'h1_arterial_pco2_min', 'h1_arterial_ph_max', 'h1_arterial_ph_min', 'h1_arterial_po2_max', 'h1_arterial_po2_min', 'h1_pao2fio2ratio_max', 'h1_pao2fio2ratio_min', 'apache_4a_hospital_death_prob', 'apache_4a_icu_death_prob', 'aids', 'cirrhosis', 'diabetes_mellitus', 'hepatic_failure', 'immunosuppression', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis', 'apache_3j_bodysystem', 'apache_2_bodysystem']]
        
    to_drop = ['h1_bilirubin_max', 'h1_bilirubin_min', 'h1_lactate_min', 'h1_lactate_max', 'h1_albumin_min', 'h1_albumin_max', 'h1_pao2fio2ratio_min', 'h1_pao2fio2ratio_max', 'h1_arterial_ph_max', 'h1_arterial_ph_min', 'h1_hco3_max', 'h1_hco3_min', 'h1_arterial_pco2_max', 'h1_arterial_pco2_min', 'h1_wbc_min', 'h1_wbc_max', 'h1_arterial_po2_min', 'h1_arterial_po2_max', 'h1_calcium_max', 'h1_calcium_min', 'h1_platelets_min', 'h1_platelets_max', 'h1_bun_max', 'h1_bun_min', 'h1_creatinine_min', 'h1_creatinine_max', 'h1_diasbp_invasive_min', 'h1_diasbp_invasive_max', 'h1_sysbp_invasive_max', 'h1_sysbp_invasive_min', 'h1_mbp_invasive_min', 'h1_mbp_invasive_max', 'h1_hematocrit_max', 'h1_hematocrit_min', 'h1_hemaglobin_max', 'h1_hemaglobin_min', 'h1_sodium_max', 'h1_sodium_min', 'h1_potassium_min', 'h1_potassium_max', 'paco2_for_ph_apache', 'ph_apache', 'paco2_apache', 'pao2_apache', 'fio2_apache', 'd1_lactate_max', 'd1_lactate_min', 'd1_diasbp_invasive_min', 'd1_diasbp_invasive_max', 'd1_sysbp_invasive_min', 'd1_sysbp_invasive_max', 'd1_mbp_invasive_max', 'd1_mbp_invasive_min', 'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min', 'd1_arterial_ph_max', 'd1_arterial_ph_min', 'd1_arterial_pco2_max', 'd1_arterial_pco2_min', 'd1_arterial_po2_max', 'd1_arterial_po2_min', 'bilirubin_apache', 'h1_inr_min', 'd1_inr_max', 'd1_inr_min', 'h1_inr_max', 'albumin_apache', 'd1_bilirubin_min', 'd1_bilirubin_max', 'h1_glucose_max', 'h1_glucose_min', 'd1_albumin_max', 'd1_albumin_min', 'urineoutput_apache', 'encounter_id', 'hospital_admit_source', 'icu_admit_source', 'icu_id', 'icu_stay_type', 'patient_id', 'hospital_id', 'readmission_status', 'weight', 'hospital_death', 'apache_4a_hospital_death_prob',	'apache_4a_icu_death_prob']

    data_total = data.drop(columns = to_drop)    
    data1 = data_total.drop(columns=['ethnicity', 'gender', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem'])
    data2 = {
            'ethnicity' : data['ethnicity'],
            'gender' : data['gender'],
            'icu_type' : data['icu_type'],
            'apache_3j_bodysystem' : data['apache_3j_bodysystem'],
            'apache_2_bodysystem' : data['apache_2_bodysystem'],
        }    
    
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)

    encoded_df2 = encode(df2)

    data1.reset_index(drop=True, inplace = True)
    encoded_df2.reset_index(drop=True, inplace = True)

    data_final = pd.concat([data1, encoded_df2], axis=1)
    pred = get_prediction(data_final, model)

    st.write(f'{pred}')
    
