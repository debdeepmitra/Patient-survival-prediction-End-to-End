import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

encoder = joblib.load(r'Models/encoder.joblib')

def encode(input_value):
  columns_to_encode = ['ethnicity', 'gender', 'icu_type', 'apache_3j_bodysystem', 'apache_2_bodysystem']
  encoded_columns = encoder.transform(input_value[columns_to_encode])
  encoded_input = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(columns_to_encode))
  return encoded_input

def get_prediction(data, model):
  val = model.predict(data)
  if val < 0.5:
    return 'It is less likely that the patient will survive! 🥺'
  elif val >= 0.5:
    return 'Chances are high that the patient will survive! 😮‍💨'
