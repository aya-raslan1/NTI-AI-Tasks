import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, SimpleRNN


dataset = np.array([5, 2, 7, 9, 16, 25])

x_train = np.array([[5, 2], [2, 7], [7, 9], [9, 16]])
y_train = np.array([7, 9, 16, 25])

x_test = np.array([[16, 25]])

model = Sequential()
model.add(SimpleRNN(units=250, input_shape=(2, 1), activation='relu'))
#model.add(LSTM(units=250, input_shape=(2, 1), activation='relu'))
model.add(Dense(units=250, activation='relu'))
model.add(Dense(units=1))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=0)

prediction = model.predict(x_test)
print(np.round(np.squeeze(prediction)))
