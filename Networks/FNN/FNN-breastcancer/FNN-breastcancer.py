import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt
from utils.swish import swish

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.utils import get_custom_objects

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import configparser

# loeme sisse millist tüüpi aktivaatoralgoritmi uurime
config = configparser.ConfigParser()
config.read("../../config.ini")
run_type = config['DEFAULT']['network_type']
run_amount = int(config['DEFAULT']['run_amount'])

# valmistame ette "swish" aktivaatorfunktsiooni
get_custom_objects().update({'swish': Activation(swish)})

# treenime x võrku, igaühe kohta teeme joonise
for i in range(run_amount):
    # andmete sisselugemine
    df = pd.read_csv("../../../Datasets/breast-cancer-wisconsin/data.csv")
    df = df.iloc[:,:-1]
    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values

    # teeme M ja B numbriteks treenimise jaoks
    labelencoder_X_1 = LabelEncoder()
    y = labelencoder_X_1.fit_transform(y)

    # jaotame andmed treeningandmeteks ja testimisandmeteks
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

    # feature scaling (puudub eestikeelne vaste)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # mudeli ehitamine
    model = Sequential()
    model.add(Dense(256, input_dim=30))
    model.add(Dense(128, activation=run_type))
    model.add(Dense(64, activation=run_type))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation=run_type))

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), callbacks=[es_callback])

    loss, accuracy = model.evaluate(X_test, y_test)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

    plt.clf()
    plt.ylim(0.8, 1.0)
    plt.xlim(0, 25)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(run_type.capitalize() + " aktivaatoralgoritm")
    plt.legend(['Treening', 'Test'], loc='upper right')
    plt.savefig("./"+run_type+"/"+str(i)+".png")
