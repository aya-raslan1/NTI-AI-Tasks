from keras.utils import to_categorical
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = Sequential()

#convolution
model.add(Conv2D(filters=128, kernel_size=(7, 7), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())

model.summary()
#Neural Network
model.add(Dense(units=250, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.1))
model.add(Dense(units=250, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.1))
model.add(Dense(units=250, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.1))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='mse', optimizer='adam', metrics='accuracy')
model.fit(x=x_test, y=y_test, batch_size=500, epochs=10)

loss, acc = model.evaluate(x=x_test, y=y_test)
print('Accuracy = ', acc)
