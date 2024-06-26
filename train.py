# -*- coding: utf-8 -*-
"""project4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16WLyNNrN5wVO13gUNJNWYUyNAFQZ6HTE
"""

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/ML_Workspace/Projects/Project-4: Patient Survival Prediction/Dataset.csv')
pd.set_option('display.max_columns', None)
df.head()

print(list(df.columns))

df.shape

df.info(verbose=True)

null_counts = df.isnull().sum()
sorted_null_counts = null_counts.sort_values(ascending=False)
for column, count in sorted_null_counts.items():
    print(f"{column}: {count}")

"""*Note: There are lots of missing values. The threshold is choosen to be 25K. So if a column has more than 25K missing values, we will drop it*"""

to_drop=[]

for column, count in sorted_null_counts.items():
  if count >= 25000:
    to_drop.append(column)

len(to_drop)

"""*Additionally, 'encounter_id', 'hospital_admit_source', 'icu_admit_source', 'icu_id', 'icu_stay_type', 'patient_id', 'hospital_id', 'readmission_status', these columns can also be dropped*"""

to_drop.extend(['encounter_id', 'hospital_admit_source', 'icu_admit_source', 'icu_id', 'icu_stay_type', 'patient_id', 'hospital_id', 'readmission_status'])

df.drop(columns=to_drop, inplace=True)

df.shape

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
sns.set()
sns.heatmap(df[['bmi', 'height', 'weight']].corr(), annot=True, cmap=plt.cm.CMRmap_r)
plt.show()

"""*Since 'bmi' and 'weight' are much co-related, we choose to drop them*"""

df.drop(columns='weight', inplace=True)

df.shape

df.info(verbose=True)

categorical_feature = [cols for cols in df.columns if df[cols].dtypes=='O']
print(categorical_feature)

df_1 = df.drop(columns=['hospital_death', 'apache_4a_hospital_death_prob',	'apache_4a_icu_death_prob'])
numerical_feature = [cols for cols in df_1.columns if df[cols].dtypes!='O']
print(len(numerical_feature))

"""**NULL value replacement:
mode for categorical,
mean for numerical**
"""

for column in categorical_feature:
    mode_value = df[column].mode()[0]  # Find the mode
    df[column].fillna(mode_value, inplace=True)

for column in numerical_feature:
    mean_value = df[column].mean()  # Find the mode
    df[column].fillna(mean_value, inplace=True)

X = df.drop(columns=['hospital_death', 'apache_4a_hospital_death_prob',	'apache_4a_icu_death_prob'])
y = df['hospital_death']

print(X.shape)
print(y.shape)

# splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state = 42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

"""**Scaling & Encoding the train data**"""

#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

#scaler = MinMaxScaler()
encoder = OneHotEncoder(sparse_output=False,handle_unknown='ignore')

#columns_to_standardize = numerical_feature
columns_to_encode = categorical_feature

#standardize_columns = scaler.fit_transform(X_train[columns_to_standardize])
#standardize_df = pd.DataFrame(standardize_columns)

#print(type(standardize_df))
#print(standardize_df.shape)

encoded_columns = encoder.fit_transform(X_train[columns_to_encode])
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(columns_to_encode))

print(type(encoded_df))
print(encoded_df.shape)
#X_test_std = scaler.transform(X_test[columns_to_standardize])

X_train_std_encoded = pd.concat([X_train[numerical_feature], encoded_df], axis=1)

X_train_std_encoded.shape

"""**Scaling & Encoding the test data**"""

#standardize_columns = scaler.transform(X_test[columns_to_standardize])
#standardize_df = pd.DataFrame(standardize_columns)

#print(type(standardize_df))
#print(standardize_df.shape)

encoded_columns = encoder.transform(X_test[columns_to_encode])
encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(columns_to_encode))

print(type(encoded_df))
print(encoded_df.shape)

X_test_std_encoded = pd.concat([X_test[numerical_feature], encoded_df], axis=1)

X_test_std_encoded.shape

"""**Building the Neural Network**"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras

num_features = len(X_train_std_encoded.columns)

model = Sequential()

model.add(Dense(512, input_shape=(num_features,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )

### Training the model
epochs = 20
batch_size = 32
history = model.fit(X_train_std_encoded, y_train, validation_data = (X_test_std_encoded, y_test), epochs=epochs, batch_size=batch_size)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Training accuracy')
plt.plot(epochs, val_acc, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

"""**Saving the model and encoder**"""

import joblib

joblib.dump(encoder,'encoder.joblib')
joblib.dump(model,'model.joblib')

print(categorical_feature)

print(numerical_feature)

#numerical_feature.index('hospital_death')

(numerical_feature)

X_train[numerical_feature].columns

describe_stats = X_train[numerical_feature].describe()

describe_stats.loc[['min', 'max']]

for cols in categorical_feature:
  print(f'Columns: ',cols)
  print(df[cols].unique())
  print('\n')