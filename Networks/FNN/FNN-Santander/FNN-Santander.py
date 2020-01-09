

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


df_train = pd.read_csv("../../../Datasets/Santander_Customer/train.csv")
df_test = pd.read_csv("../../../Datasets/Santander_Customer/test.csv")

target_train = df_train.pop('target')
df_train = df_train.drop(columns=['ID_code'])
data_train = tf.data.Dataset.from_tensor_slices((df_train.values, target_train.values))
data_train = data_train.shuffle(len(df_train)).batch(50)

data_test = df_test.to_numpy()


model = Sequential()
model.add(Dense(64, input_dim=200))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

#es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

model.fit(data_train, epochs=10)
