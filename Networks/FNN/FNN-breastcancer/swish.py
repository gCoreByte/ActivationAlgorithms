import pandas as pd
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from utils.swish import swish
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import get_custom_objects


df = pd.read_csv("../../../Datasets/breast-cancer-wisconsin/data.csv")
df = df.iloc[:,:-1]
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

get_custom_objects().update({'swish': Activation(swish)})

model = Sequential()
model.add(Dense(256, input_dim=30))
model.add(Dropout(0.2))
model.add(Dense(128, activation='swish'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='swish'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='swish'))

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[es_callback])

loss, accuracy = model.evaluate(X_test, y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("swish.png")
plt.show()
