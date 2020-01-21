import matplotlib.pyplot as plt
import numpy as np

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def swish(x):
    beta = 1
    return x*sigmoid(beta*x)

def elu(x):
    alpha = 1
    if x >= 0:
        return x
    else:
        return alpha*(np.exp(x)-1)

#--------------------------------#
plt.clf()
X = np.linspace(-10,10,50)
Y = [relu(x) for x in X]
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("RELU funktsioon")
plt.grid(True)
plt.plot(X,Y)
plt.savefig("relu_graph.png")
#--------------------------------#
plt.clf()
X = np.linspace(-10,10,50)
Y = [sigmoid(x) for x in X]
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Sigmoidfunktsioon")
plt.grid(True)
plt.plot(X,Y)
plt.savefig("sigmoid_graph.png")
#--------------------------------#
plt.clf()
X = np.linspace(-10,10,50)
Y = [swish(x) for x in X]
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.title("Swish funktsioon")
plt.grid(True)
plt.plot(X,Y)
plt.savefig("swish_graph.png")
#--------------------------------#
plt.clf()
X = np.linspace(-5,5,50)
Y = [elu(x) for x in X]
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.yticks(np.arange(-1, 6, 1))
plt.title("ELU funktsioon")
plt.grid(True)
plt.plot(X,Y)
plt.savefig("elu_graph.png")