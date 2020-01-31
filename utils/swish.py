from tensorflow.keras import backend as K


def swish(x, beta=1.0):
    return x * K.sigmoid(beta * x)
