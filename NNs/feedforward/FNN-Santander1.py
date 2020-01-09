import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow_core.python.keras.layers.core import Dense, Dropout


#df_test = pd.read_csv('../../Datasets/Santander_Customer/test.csv')

df = pd.read_csv('../../Datasets/Santander_Customer/train.csv')

# split data
df_train = df[:150000]
df_test = df[150000:]
# training data
target_train = df_train.pop("target")
df_train = df_train.drop(columns=["ID_code"])
dataset_train = tf.data.Dataset.from_tensor_slices((df_train.values, target_train.values))

# testing data

target_test = df_test.pop("target")
df_test = df_test.drop(columns=["ID_code"])
dataset_test = tf.data.Dataset.from_tensor_slices((df_test.values, target_test.values))

#dataset_train = dataset_train.shuffle
#dataset_test = dataset_test.shuffle
#for features_tensor, target_tensor in dataset_train:
#    print(f'features:{features_tensor} target:{target_tensor}')
dataset_train = dataset_train.shuffle(150000).batch(200)
dataset_test = dataset_test.shuffle(50000).batch(200)


model = tf.keras.models.Sequential()
model.add(Dense(20, input_dim=200, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.25))
# final layer, binary classification -> sigmoid activation works
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(dataset_train, epochs=10)

loss, accuracy = model.evaluate(dataset_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)