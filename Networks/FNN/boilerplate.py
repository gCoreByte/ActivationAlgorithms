import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# loeme sisse andmed
df = pd.read_csv("")
# eristame treeningandmed ja oodatavad väärtused, X ja y vastavalt
X = df.iloc[:, 2:].values
y = df.iloc[:, 1].values

# LabelEncoder() võimaldab meil muuta M ja B numbriteks algoritmi jaoks
labelencoder_X_1 = LabelEncoder()
y = labelencoder_X_1.fit_transform(y)

# eraldame  10% andmetest testimiseks
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# ühtlustame andmeid, et oleks paremad tulemused

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# loome mudeli
model = Sequential()
model.add(Dense(256, input_dim=30))
model.add(Dense(128, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(1, activation='sigmoid'))

# kui test andmete täpsus väheneb, lõpetame treenimise varem (overfitting)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

#treenime mudelit
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[es_callback])

# kontrollime täpsust
loss, accuracy = model.evaluate(X_test, y_test)

print("Loss: ", loss)
print("Accuracy: ", accuracy)