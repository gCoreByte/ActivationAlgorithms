import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation
from tensorflow.keras.utils import get_custom_objects

from utils.swish import swish

import configparser

config = configparser.ConfigParser()
config.read("../../config.ini")
run_type = config['DEFAULT']['network_type']
run_amount = int(config['DEFAULT']['run_amount'])

get_custom_objects().update({'swish': Activation(swish)})

acc_all = []
len_all = 0

# loome mudeli
for i in range(run_amount):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory("../../../Datasets/chest_xray/train",
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

    validation_set = test_datagen.flow_from_directory("../../../Datasets/chest_xray/val",
                                                      target_size=(64, 64),
                                                      batch_size=32,
                                                      class_mode='binary')

    test_set = test_datagen.flow_from_directory("../../../Datasets/chest_xray/test",
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    model = Sequential()

    #model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
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

    plt.clf()
    plt.ylim(0.8, 1.0)
    plt.xlim(0, 25)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(run_type.capitalize() + " aktivaatoralgoritm")
    plt.legend(['Treening täpsus', 'Testimise täpsus'], loc='upper right')
    plt.savefig("./" + run_type + "/1" + str(i) + ".png")

plt.clf()
plt.ylim(0.5, 1.1)
X = [x for x in range(len(acc_all))]
plt.plot(X, acc_all)
plt.axhline(1, color='red')
plt.legend(['Võrkude täpsus'])
plt.text(1, 0.55, "Keskmine täpsus:" + str(round(sum(acc_all)/len(acc_all), 3)), fontsize=11)
plt.text(1, 0.6, "Keskmine treeningupikkus:" + str(round(len_all/len(acc_all), 3)), fontsize=11)
plt.savefig(run_type+"_avg_acc.png")