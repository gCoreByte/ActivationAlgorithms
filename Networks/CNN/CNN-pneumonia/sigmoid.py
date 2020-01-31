import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

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

model.add(Conv2D(32, (7, 7), activation='sigmoid', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# kui test andmete täpsus väheneb 3 korda järjest, lõpetame treenimise varem (aitab vältida ülesobitavust)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# kompileerime ja treenime mudelit
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit_generator(training_set, epochs=3, steps_per_epoch=163, validation_data=validation_set, validation_steps=312, callbacks=[es_callback])

# kontrollime täpsust
loss, accuracy = model.evaluate_generator(validation_set)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

#print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
plt.savefig("sigmoid.png")