import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df_train = pd.read_csv("../../../Datasets/Dont_Overfit/train.csv", index_col='id')
df_test = pd.read_csv("../../../Datasets/Dont_Overfit/test.csv", index_col='id')

target_train = df_train.pop('target')
data_train = tf.data.Dataset.from_tensor_slices((df_train.values, target_train.values))
data_train = data_train.shuffle(len(df_train)).batch(1)

data_test = df_test.to_numpy()
#data_test = tf.data.Dataset.from_tensor_slices(df_train.values)
#data_test = data_test.shuffle(len(df_test))

model = Sequential()
model.add(Dense(64, input_dim=300))
model.add(Dense(32, activation='sigmoid'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))

es_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(data_train, epochs=2, callbacks=[es_callback])

sol_test = model.predict_classes(data_test)
print(sol_test)

x = 250
y = 0

file = open("../solutions/FNN-overfit/sol-overfit-sigmoid.csv", "w")
file.write("id,target\n")
for y in sol_test:
    file.write(str(x) + "," + str(y).strip("[]") + "\n")
    x += 1
file.close()

# Avg -