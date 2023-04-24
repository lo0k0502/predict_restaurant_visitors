import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

num_timesteps = 365

def preprocess():
    reservation_data = pd.read_csv('processed_data/reservation_data.csv')
    visit_data = pd.read_csv('processed_data/visit_data.csv')
    store_data = pd.read_csv('processed_data/store_data.csv')
    date_info = pd.read_csv('date_info.csv')

    date_info.drop('day_of_week', axis=1, inplace=True)
    date_info.rename(columns={ 'calendar_date': 'visit_date' }, inplace=True)

    merged = pd.merge(reservation_data, visit_data, on=['store_id', 'visit_date'])
    merged = pd.merge(merged, store_data, on='store_id')
    merged = pd.merge(merged, date_info, on='visit_date')

    le = LabelEncoder()

    merged['visit_year'] = pd.to_datetime(merged['visit_date']).dt.year
    merged['visit_month'] = pd.to_datetime(merged['visit_date']).dt.month
    merged['visit_day'] = pd.to_datetime(merged['visit_date']).dt.day
    merged['visit_weekday'] = pd.to_datetime(merged['visit_date']).dt.weekday
    merged['store_id'] = le.fit_transform(merged['store_id'])

    merged.drop('visit_date', axis=1, inplace=True)
    order = [
        'store_id',
        'visit_year',
        'visit_month',
        'visit_day',
        'visit_weekday',
        'holiday_flg',
        'latitude',
        'longitude',
        'use_hpg',
        'reserve_visitors',
        'visitors',
    ]
    preprocessed_data = merged.loc[:, order]
    
    train_data = preprocessed_data[preprocessed_data['visit_year'] == 2016]
    test_data = preprocessed_data[preprocessed_data['visit_year'] == 2017]

    X_train = train_data.drop('visitors', axis=1).values
    Y_train = train_data['visitors'].values
    X_test = test_data.drop('visitors', axis=1).values
    Y_test = test_data['visitors'].values

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    num_features = X_train.shape[1]
    num_samples = len(X_train) - num_timesteps + 1

    X = np.zeros((num_samples, num_timesteps, num_features))
    Y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        X[i] = X_train[i : i + num_timesteps].reshape((-1, num_features))
        Y[i] = Y_train[i + num_timesteps - 1]
    X_train, Y_train = X, Y

    num_samples = len(X_test) - num_timesteps + 1
    X = np.zeros((num_samples, num_timesteps, num_features))
    Y = np.zeros((num_samples, 1))
    for i in range(num_samples):
        X[i] = X_test[i : i + num_timesteps].reshape((-1, num_features))
        Y[i] = Y_test[i + num_timesteps - 1]
    X_test, Y_test = X, Y

    np.save('processed_data/X_train', X_train)
    np.save('processed_data/Y_train', Y_train)
    np.save('processed_data/X_test', X_test)
    np.save('processed_data/Y_test', Y_test)