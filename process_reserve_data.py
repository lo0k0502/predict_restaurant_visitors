import pandas as pd
import numpy as np

def process_reserve_data():
    air_reserve = pd.read_csv('air_reserve.csv')
    hpg_reserve = pd.read_csv('hpg_reserve.csv')
    store_id_relation = pd.read_csv('store_id_relation.csv')

    merged_air_reserve = pd.merge(air_reserve, store_id_relation, on='air_store_id', how='outer')
    merged_hpg_reserve = pd.merge(hpg_reserve, store_id_relation, on='hpg_store_id', how='outer')

    merged = pd.merge(merged_air_reserve, merged_hpg_reserve, how='outer')

    merged['visit_date'] = pd.to_datetime(merged['visit_datetime']).dt.date

    air_reservations_by_date = merged.groupby(['air_store_id', 'visit_date'])['reserve_visitors'].sum().reset_index()
    air_reservations_by_date['reserve_visitors'] = air_reservations_by_date['reserve_visitors'].astype(int)

    reservation_data = pd.merge(air_reservations_by_date, store_id_relation, on='air_store_id', how='outer')
    reservation_data.loc[reservation_data['hpg_store_id'].isnull(), 'store_id'] = reservation_data['air_store_id']
    reservation_data.loc[reservation_data['hpg_store_id'].notnull(), 'store_id'] = reservation_data['air_store_id'] + reservation_data['hpg_store_id']

    drop_columns = ['air_store_id', 'hpg_store_id']
    reservation_data.drop(drop_columns, axis=1, inplace=True)
    reservation_data = reservation_data.loc[:, ['store_id', 'visit_date', 'reserve_visitors']]

    reservation_data.to_csv('processed_data/reservation_data.csv', index=False)