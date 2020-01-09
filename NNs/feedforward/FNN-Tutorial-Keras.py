from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense

dataset = genfromtxt("pima-indians-diabetes.data.csv", delimiter=",")

X = dataset[:, 0:8]
y = dataset[:, 8]

model = Sequential()
model.add(Dense(500, input_dim=8, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=15, batch_size=5)
_, accuracy = model.evaluate(X, y)
print(accuracy)

predictions = model.predict_classes(X)
for i in range(5):
    print("%s - %d, expected %d" % (X[i].tolist(), predictions[i], y[i]))
