import pandas as pd

def process_store_data():
    air_store_info = pd.read_csv('air_store_info.csv')
    hpg_store_info = pd.read_csv('hpg_store_info.csv')
    store_id_relation = pd.read_csv('store_id_relation.csv')

    merged_air_store_info = pd.merge(air_store_info, store_id_relation, on='air_store_id', how='outer')
    merged_air_store_info.drop(['air_genre_name', 'air_area_name'], axis=1, inplace=True)
    merged_hpg_store_info = pd.merge(hpg_store_info, store_id_relation, on='hpg_store_id', how='outer')
    merged_hpg_store_info.drop(['hpg_genre_name', 'hpg_area_name'], axis=1, inplace=True)
    merged_hpg_store_info.dropna(subset=['latitude', 'longitude'], inplace=True)

    merged = pd.merge(merged_air_store_info, merged_hpg_store_info, how='outer')
    merged.dropna(subset=['air_store_id'], inplace=True)

    merged.loc[merged['hpg_store_id'].isnull(), 'use_hpg'] = 0
    merged.loc[merged['hpg_store_id'].notnull(), 'use_hpg'] = 1
    merged['use_hpg'] = merged['use_hpg'].astype(int)
    merged.loc[merged['hpg_store_id'].isnull(), 'store_id'] = merged['air_store_id']
    merged.loc[merged['hpg_store_id'].notnull(), 'store_id'] = merged['air_store_id'] + merged['hpg_store_id']

    drop_columns = ['air_store_id', 'hpg_store_id']
    merged.drop(drop_columns, axis=1, inplace=True)
    merged = merged.loc[:, ['store_id', 'use_hpg', 'latitude', 'longitude']]

    merged.to_csv('processed_data/store_data.csv', index=False)