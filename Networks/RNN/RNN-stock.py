import gc

import pandas as pd
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from utils.swish import swish

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras.utils import get_custom_objects

import configparser

config = configparser.ConfigParser()
config.read("../config.ini")
run_type = config['DEFAULT']['network_type']
stock = config['DEFAULT']['stock_name']
run_amount = int(config['DEFAULT']['run_amount'])

get_custom_objects().update({'swish': Activation(swish)})

data = pd.read_csv("../../Datasets/stock-data" + stock)


# create training inputs
window_len = 10

# Create a data point (i.e. a date) which splits the training and testing set
split_date = list(data["Date"][-(2 * window_len + 1):])[0]

# Split the training and test set
training_set, test_set = data[data['Date'] < split_date], data[data['Date'] >= split_date]
training_set = training_set.drop(['Date', 'OpenInt'], 1)
test_set = test_set.drop(['Date', 'OpenInt'], 1)

# Create windows for training
LSTM_training_inputs = []
for i in range(len(training_set) - window_len):
    temp_set = training_set[i:(i + window_len)].copy()

    for col in list(temp_set):
        temp_set[col] = temp_set[col] / temp_set[col].iloc[0] - 1

    LSTM_training_inputs.append(temp_set)
LSTM_training_outputs = (training_set['Close'][window_len:].values / training_set['Close'][:-window_len].values) - 1

LSTM_training_inputs = [np.array(LSTM_training_input) for LSTM_training_input in LSTM_training_inputs]
LSTM_training_inputs = np.array(LSTM_training_inputs)

# Create windows for testing
LSTM_test_inputs = []
for i in range(len(test_set) - window_len):
    temp_set = test_set[i:(i + window_len)].copy()

    for col in list(temp_set):
        temp_set[col] = temp_set[col] / temp_set[col].iloc[0] - 1

    LSTM_test_inputs.append(temp_set)
LSTM_test_outputs = (test_set['Close'][window_len:].values / test_set['Close'][:-window_len].values) - 1

LSTM_test_inputs = [np.array(LSTM_test_inputs) for LSTM_test_inputs in LSTM_test_inputs]
LSTM_test_inputs = np.array(LSTM_test_inputs)

loss = []
for i in range(run_amount):
    model = Sequential()
    model.add(LSTM(32, input_shape=(10, 5)))
    model.add(Dropout(0.1))
    model.add(Dense(16, activation=run_type))
    model.add(Dense(1, activation=run_type))

    model.compile(optimizer='adam', loss='mae')

    history = model.fit(LSTM_training_inputs, LSTM_training_outputs, epochs=25, batch_size=32)

    loss.append(mean_absolute_error(LSTM_test_outputs, model.predict(LSTM_test_inputs)))

    plt.clf()
    plt.ylim(-0.06, 0.12)
    plt.title(run_type.capitalize() + " aktivaatoralgoritm")
    plt.plot(LSTM_test_outputs, label = "Reaalne väärtus")
    plt.plot(model.predict(LSTM_test_inputs), label = "Ennustatud väärtus")
    plt.legend()
    plt.savefig("./"+run_type+"/"+str(i)+".png")

plt.clf()
plt.ylim(0, 0.05)
X = [x for x in range(len(loss))]
plt.plot(X, loss)
plt.axhline(1, color='red')
plt.legend(['Võrkude keskmine absoluutne viga'])
plt.title(run_type.capitalize() + " aktivaatoralgoritm")
plt.legend(['Keskmine absoluutne viga'], loc='upper left')
plt.text(0, 0.04, "Keskmise absoluutse vea keskmine: " + str(round(sum(loss)/len(loss), 3)), fontsize=11)
plt.savefig(run_type+"_avg_acc.png")