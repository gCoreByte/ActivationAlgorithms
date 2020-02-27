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

get_custom_objects().update({'swish': Activation(swish)})

# laeme treeningpildid ära
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("../../../Datasets/chest_xray/train",
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

validation_set = test_datagen.flow_from_directory("../../../Datasets/chest_xray/val",
                                                  target_size = (64, 64),
                                                  batch_size = 32,
                                                  class_mode = 'binary')

test_set = test_datagen.flow_from_directory("../../../Datasets/chest_xray/test",
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode = 'binary')



# loome mudeli

model = Sequential()

model.add(Conv2D(32, (3, 3), activation=run_type, input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation=run_type))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=run_type))
model.add(Dense(1, activation=run_type))

# kui test andmete täpsus väheneb 3 korda järjest, lõpetame treenimise varem (aitab vältida ülesobitavust)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# kompileerime ja treenime mudelit
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#
# cnn = Sequential()
#
# #Convolution
# cnn.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
#
# #Pooling
# cnn.add(MaxPooling2D(pool_size = (2, 2)))
#
# # 2nd Convolution
# cnn.add(Conv2D(32, (3, 3), activation="relu"))
#
# # 2nd Pooling layer
# cnn.add(MaxPooling2D(pool_size = (2, 2)))
#
# # Flatten the layer
# cnn.add(Flatten())
#
# # Fully Connected Layers
# cnn.add(Dense(activation = 'relu', units = 128))
# cnn.add(Dense(activation = run_type, units = 1))
#
# # Compile the Neural network
# cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# cnn.summary()
history = model.fit_generator(training_set, epochs=100, steps_per_epoch=163, validation_data=validation_set, validation_steps=163, callbacks=[es_callback])

# kontrollime täpsust
loss, accuracy = model.evaluate_generator(validation_set)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

#print(history.history.keys())
plt.ylim(0.8, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title(run_type.capitalize() + " aktivaatoralgoritm")
plt.legend(['Treening', 'Test'], loc='upper left')
plt.show()
plt.savefig(run_type + ".png")