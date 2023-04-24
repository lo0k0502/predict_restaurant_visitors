import numpy as np
from keras import models
from keras.metrics import mean_squared_logarithmic_error

from model import rmsle

def predict():
    X_test = np.load('processed_data/X_test.npy')
    Y_test = np.load('processed_data/Y_test.npy')

    model = models.load_model('model.h5', custom_objects={ 'rmsle': rmsle })

    Y_pred = model.predict(X_test)
    if len(Y_pred.shape) == 3: Y_pred = np.squeeze(Y_pred, 2)
    Y_pred = np.round(np.mean(Y_pred, axis=1)).reshape(Y_test.shape)

    accuracy = np.mean(np.sqrt(mean_squared_logarithmic_error(Y_test, Y_pred)))

    np.savetxt('processed_data/Y_pred.csv', Y_pred, delimiter=',', fmt='%f')
    print('Y_test: ', Y_test.shape)
    print(Y_test)
    print('Y_pred: ', Y_pred.shape)
    print(Y_pred)
    print('accuracy: ', accuracy)