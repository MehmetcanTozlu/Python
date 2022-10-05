import math
import matplotlib.pyplot as plt
import numpy as np

# Linear Function
def linearFunction(x):
    fx = []
    for i in x:
        fx.append(i)
    return fx

# Step Function
def stepFunction(x):
    fx = []
    for i in x:
        if i < 0:
            fx.append(0)
        else:
            fx.append(1)
    return fx

# Sigmoid Function
def sigmoidFunction(x):
    fx = []
    for i in x:
        fx.append(1 / (1 + math.exp(-i)))
    return fx

# Hyperbolic Tangent Function
def tanh(x):
    fx = []
    for i in x:
        fx.append((math.exp(i) - math.exp(-i)) / (math.exp(i) + math.exp(-i)))
    return fx

# ReLU
def relu(x):
    fx = []
    for i in x:
        if i < 0:
            fx.append(0)
        else:
            fx.append(i)
    return fx

# Leaky ReLU
def lRelu(x):
    fx = []
    for i in x:
        if i >= 0:
            fx.append(i)
        else:
            fx.append(i/10)
    return fx

# Araligimiz
arr = np.arange(-3., 3., 0.1)

# Nesne Uretme Islemlerimiz
lin = linearFunction(arr)
step = stepFunction(arr)
sigmoid = sigmoidFunction(arr)
tanh = tanh(arr)
relu = relu(arr)
lrelu = lRelu(arr)
swish = arr * sigmoid

# Grafiklerin Cizdirilmesi
line_1 = plt.plot(arr, lin, label='Linear')
line_2 = plt.plot(arr, step, label='Step')
plt.show()

line_3 = plt.plot(arr, sigmoid, label='Sigmoid')
line_4 = plt.plot(arr, tanh, label='TanH')
line_5 = plt.plot(arr, relu, label='ReLU')
line_6 = plt.plot(arr, lrelu, label='LReLU')
line_7 = plt.plot(arr, swish, label='Swish')

plt.legend(handles=[line_1, line_2, line_3, line_4, line_5, line_6, line_7])
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.show()
