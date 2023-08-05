import numpy as np
# These are XOR inputs
x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]) #x1 and x2
# These are XOR outputs
y = np.array([0, 1, 1, 0])
# Number of inputs
num_x = 2
# Number of neurons in output layer
num_y = 1
# Number of neurons in hidden layer
num_h = 2
# Learning rate
lr = 0.1
# Define random seed for consistent results
np.random.seed(2)
# Define weight matrices for neural network
w1 = np.random.rand(num_h, num_x)   # Weight matrix for hidden layer
w2 = np.random.rand(num_y, num_h)   # Weight matrix for output layer
#w1 = np.array([[0.5, 0.5], [5, 5]])
#w2 = np.array([[-10.0, 5]])

# I didn't use bias units

# I used sigmoid activation function for hidden layer and output
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Sigmoid derivative
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

# Forward propagation
def forward_prop(w1, w2, x):
    z1 = np.dot(w1, x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backward propagation
def back_prop(z1,z2,w1,w2,a1,a2,y):

    dz2 = (a2-y) * sigmoid_deriv(z2)
    dw2 = np.dot(dz2, a1.T)
    dz1 = np.dot(w2.T, dz2) * sigmoid_deriv(z1)
    dw1 = np.dot(dz1, x.T)
    return dz1, dw1, dz2, dw2


def predict(w1, w2, input):
    z1, a1, z2, a2 = forward_prop(w1, w2, input)
    a2 = np.squeeze(a2)
    return a2


iterations = 10000
for i in range(iterations):
    z1, a1, z2, a2 = forward_prop(w1, w2, x)
    dz1, dw1, dz2, dw2 = back_prop(z1, z2, w1, w2, a1, a2, y)
    w2 = w2-lr*dw2
    w1 = w1-lr*dw1

print ('w1---->', w1)
print ('w2---->', w2)
comp1 = np.array([[0], [0]])
pre = predict(w1, w2, comp1)
print("For input [0, 0] output is", np.round(pre))
comp2 = np.array([[0], [1]])
pre = predict(w1, w2, comp2)
print("For input [0, 1] output is", np.round(pre))
comp3 = np.array([[1], [0]])
pre = predict(w1, w2, comp3)
print("For input [1, 0] output is", np.round(pre))
comp4 = np.array([[1], [1]])
pre = predict(w1, w2, comp4)
print("For input [1, 1] output is", np.round(pre))
