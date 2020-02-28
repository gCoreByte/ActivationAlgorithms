import pandas as pd
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from utils.swish import swish


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras.utils import get_custom_objects

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import configparser

config = configparser.ConfigParser()
config.read("../config.ini")
run_type = config['DEFAULT']['network_type']
stock = config['DEFAULT']['stock_name']
run_amount = int(config['DEFAULT']['run_amount'])

get_custom_objects().update({'swish': Activation(swish)})

data = pd.read_csv("../../Datasets/stock-data" + stock)
data = data.drop(['Date', 'OpenInt'], 1)

trainAmount = round(len(data.index)/100*90)
train_data = data[:trainAmount].iloc[:,2:3].values
test_data = data[trainAmount:].iloc[:,2:3].values

sc = StandardScaler()
scaled_train_data = sc.fit_transform(train_data)

X_train = []
y_train = []

for i in range(50, len(train_data)):
    X_train.append(scaled_train_data[i-50:i, 0])
    y_train.append(scaled_train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = []
test_data = test_data.reshape(-1, 1)
for i in range(100, len(test_data)):
    X_test.append(test_data[i-50:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))



for i in range(run_amount):
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(32, activation=run_type, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(16, activation=run_type, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation=run_type))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation=run_type))

    model.compile(optimizer='adam', loss='mae')

    history = model.fit(X_train, y_train, epochs=25, batch_size=32)
    print(test_data)
    print(model.predict(X_test))
    print(sc.inverse_transform(model.predict(X_test)))
    #plt.xlim(0, 25)
    #plt.ylim(-0.06, 0.12)
    # plt.plot(test_data, label = "reaalne")
    # plt.plot(sc.inverse_transform(model.predict(X_test)), label = "ennustatud")
    # plt.legend()
    # plt.savefig("./"+run_type+"/"+str(i)+".png")

#MAE = mean_absolute_error(LSTM_test_outputs, nn_model.predict(LSTM_test_inputs))

# loss, accuracy = model.evaluate(lstm_test_input, lstm_test_output)
#
# print("Loss: ", loss)
# print("Accuracy: ", accuracy)
#
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title(run_type.capitalize() + " aktivaatoralgoritm")
# plt.legend(['Treening', 'Test'], loc='upper left')
# plt.savefig(run_type+".png")
# plt.show()
