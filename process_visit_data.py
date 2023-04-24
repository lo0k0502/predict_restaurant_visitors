import pandas as pd

def process_visit_data():
    air_visit_data = pd.read_csv('air_visit_data.csv')
    store_id_relation = pd.read_csv('store_id_relation.csv')

    air_visit_data['visit_date'] = pd.to_datetime(air_visit_data['visit_date'])
    air_visit_data['visit_date'] = air_visit_data['visit_date'].dt.date
    air_visit_data.dropna(subset=['visitors'], inplace=True)

    air_visit_data = pd.merge(air_visit_data, store_id_relation, on='air_store_id', how='outer')
    air_visit_data.loc[air_visit_data['hpg_store_id'].isnull(), 'store_id'] = air_visit_data['air_store_id']
    air_visit_data.loc[air_visit_data['hpg_store_id'].notnull(), 'store_id'] = air_visit_data['air_store_id'] + air_visit_data['hpg_store_id']

    drop_columns = ['air_store_id', 'hpg_store_id']
    air_visit_data.drop(drop_columns, axis=1, inplace=True)
    air_visit_data = air_visit_data.loc[:, ['store_id', 'visit_date', 'visitors']]

    air_visit_data.to_csv('processed_data/visit_data.csv', index=False)