import numpy as np
from keras import backend as K, Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.metrics import mean_squared_logarithmic_error
from keras.optimizers import Adam, SGD

from preprocess import num_timesteps

def rmsle(y_true, y_pred):
    return K.sqrt(mean_squared_logarithmic_error(K.log(y_true + 1), K.log(y_pred + 1)))

def model():
    X_train = np.load('processed_data/X_train.npy')
    Y_train = np.load('processed_data/Y_train.npy')
    X_test = np.load('processed_data/X_test.npy')
    Y_test = np.load('processed_data/Y_test.npy')

    model = Sequential()
    model.add(LSTM(64, input_shape=(num_timesteps, X_train.shape[-1])))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))

    model.compile(optimizer=SGD(learning_rate=0.001), loss=rmsle)
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_split=0.1)

    test_loss = model.evaluate(X_test, Y_test)
    print('loss: ', test_loss)
    model.summary()
    model.save('model.h5')
