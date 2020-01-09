from numpy import genfromtxt
from keras.models import Sequential
from keras.layers import Dense

# TODO clean up data


# nahhuj autist
dataset = genfromtxt("C:\\Users\\User\\PycharmProjects\\UT\\Data\\santander-customer-transaction-prediction\\train.csv", delimiter=",")
dataset = dataset[1:]

X = dataset[:, 2:200]
y = dataset[:, 1]
dataset = dataset[1:]
model = Sequential()
model.add(Dense(20, input_dim=198, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='relu'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=2, batch_size=5)
_, accuracy = model.evaluate(X, y)
#print(accuracy)

predictions = model.predict_classes(X)
for i in range(50):
    print("%s - %d, expected %d" % (X[i].tolist(), predictions[i], y[i]))
