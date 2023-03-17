# utils.py

import os
import pandas as pd
import numpy as np


def get_locations(DATA_PATH):
    'This function returns a list of locations for which we have data.'
    locations = [location for location in os.listdir(f'{DATA_PATH}/power') if location.endswith('_power.csv')]
    locations = [location.split('_')[0] for location in locations]
    return locations


def load_data(DATA_PATH, location:str, data_type:str):
    'This function loads the data from the data folder given the location and the data type.'
    df = pd.read_csv(f'{DATA_PATH}/{data_type}/{location}_{data_type}.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def drop_duplicate_index(df):
    'This function drops duplicate indices from a dataframe.'
    df = df[~df.index.duplicated(keep='first')]
    return df


def infer_frequency(df):
    'This function infers the frequency of the time series data.'
    freq = pd.infer_freq(df.index)

    if freq is None:
        #taking the mode of the difference between the indices
        freq = df.index.to_series().diff().mode()[0]
    return freq

def ts_list_concat(ts_list):
    '''This function concatenates a list of time series into one time series'''
    ts = ts_list[0]
    for i in range(1, len(ts_list)):
        ts = ts.append(ts_list[i])
    return ts


def make_index_same(ts1, ts2):
    '''This function makes the indices of two time series the same'''
    ts1 = ts1[ts2.start_time():ts2.end_time()]
    ts2 = ts2[ts1.start_time():ts1.end_time()]
    return ts1, ts2


def get_df_compares_list(historics, gt):
    '''Returns a list of dataframes with the ground truth and the predictions next to each other'''

    df_gt = gt.pd_dataframe()
    df_compare_list = []
    for ts in historics:
        if ts.is_probabilistic:
            df = ts.quantile_df(0.5)
        else:
            df = ts.pd_dataframe()
        
        df['gt'] = df_gt

        df.reset_index(inplace=True)
        df = df.iloc[:,1:]
        df_compare_list.append(df)

    return df_compare_list
        
def get_df_diffs(df_list):
        '''Returns a dataframe with the differences between the first column and the rest of the columns'''
    
        df_diffs = pd.DataFrame(index=range(df_list[0].shape[0]))
        for df in df_list:
            df_diff = df.copy()
            diff = (df_diff.iloc[:,0].values - df_diff.iloc[:,1]).values
            df_diffs = pd.concat([df_diffs, pd.DataFrame(diff)], axis=1)
        return df_diffs


def train_val_test_split(ts_list, train_end, val_end):
    '''This function splits the time series into train, validation and test sets
    ts_list: list of time series
    train_end: end of the training set
    val_end: end of the validation set'''

    ts_train_list = []
    ts_val_list = []
    ts_test_list = []

    for ts in ts_list:
        ts_train = ts[:train_end]
        ts_val = ts[train_end:val_end]
        ts_test = ts[val_end:]
        ts_train_list.append(ts_train)
        ts_val_list.append(ts_val)
        ts_test_list.append(ts_test)
    
    return ts_train_list, ts_val_list, ts_test_list


def train_models(models:list, ts_train_list_piped, ts_train_weather_list_piped, idx):
    '''This function trains a list of models on the training data'''
    for model in models:
        model.fit(ts_train_list_piped[idx], future_covariates=ts_train_weather_list_piped[idx])
    return models
