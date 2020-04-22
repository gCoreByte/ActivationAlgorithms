import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import csv

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.utils import get_custom_objects

from utils.swish import swish

import configparser

config = configparser.ConfigParser()
config.read("../config.ini")
run_type = config['DEFAULT']['network_type']
run_amount = int(config['DEFAULT']['run_amount'])

get_custom_objects().update({'swish': Activation(swish)})

acc_all = []
len_all = 0

file = open(run_type + ".csv", "w")
file_writer = csv.writer(file)

# loome mudeli
for i in range(run_amount):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory("../../Datasets/chest_xray/train",
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

    validation_set = test_datagen.flow_from_directory("../../Datasets/chest_xray/val",
                                                      target_size=(64, 64),
                                                      batch_size=32,
                                                      class_mode='binary')

    test_set = test_datagen.flow_from_directory("../../Datasets/chest_xray/test",
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=run_type))
    model.add(Dense(1, activation='sigmoid'))

    # kui test andmete täpsus väheneb 3 korda järjest, lõpetame treenimise varem (aitab vältida ülesobitavust)
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    # kompileerime ja treenime mudelit
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("NETWORK NR: " + str(i))
    history = model.fit(training_set, epochs=25, validation_data=validation_set, callbacks=[es_callback])

    # kontrollime täpsust
    loss, accuracy = model.evaluate(validation_set)
    acc_all.append(accuracy)
    len_all += len(history.history['loss'])

    # salvestame faili
    file_writer.writerow([accuracy, history.history['accuracy'], history.history['val_accuracy']])


file.close()
file = open(run_type + "_final.csv", "w")
file_writer = csv.writer(file)
file_writer.writerow([acc_all, len_all])
file.close()