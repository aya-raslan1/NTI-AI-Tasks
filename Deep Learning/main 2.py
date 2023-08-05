import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#_, ax = plt.subplots(ncols=4, nrows=4, figsize=(28, 28))
#ind = 0
#for i in range(4):
#    for j in range(4):
#        ax[i][j].imshow(x_train[ind])
#        ind +=1
#plt.show()

def re_shape(train, test):
    x_train_ = np.array(train).reshape(60000, 28*28)
    x_test_ = np.array(test).reshape(10000, 28*28)
    return x_train_, x_test_

def normalize(train, test):
    sc = StandardScaler()
    x_train_ = sc.fit_transform(train)
    x_test_ = sc.fit_transform(test)
    return x_train_, x_test_


x_train, x_test = re_shape(x_train, x_test)
x_train, x_test = normalize(x_train, x_test)

model = Sequential()
model.add(Dense(units=250, input_shape=(784,), activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.1))
model.add(Dense(units=250, input_shape=(784,), activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.1))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='mse', optimizer='adam', metrics='accuracy')
#model.summary()
model.fit(x=x_train, y=y_train, batch_size=500, epochs=10)
