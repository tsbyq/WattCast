# utils.py

'''This file contains utility functions that are used in the notebooks.'''

import os
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau

class LRMonitor(Callback):
    def __init__(self, monitor='val_loss', mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=False):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose

        self.scheduler = None
        self.best_score = None
        self.num_bad_epochs = None

    def on_train_start(self, trainer, pl_module):
        optimizer = trainer.optimizers[0]
        self.scheduler = ReduceLROnPlateau(optimizer, mode=self.mode, factor=self.factor, patience=self.patience, min_lr=self.min_lr, verbose=self.verbose)

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        current_score = logs[self.monitor]
        if self.best_score is None:
            self.best_score = current_score
            self.num_bad_epochs = 0
        elif self._compare_score(current_score, self.best_score):
            self.best_score = current_score
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                if self.verbose:
                    print(f'Reducing learning rate. Best score: {self.best_score}, Current score: {current_score}')
                self.scheduler.step(self.best_score)
                self.best_score = None
                self.num_bad_epochs = 0

    def _compare_score(self, current_score, best_score):
        if self.mode == 'min':
            return current_score < best_score
        else:
            return current_score > best_score






def load_from_model_artifact_checkpoint(model_class, base_path, checkpoint_path):
    model = model_class.load(base_path)
    model.model = model._load_from_checkpoint(checkpoint_path)
    return model


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
    for i in range(1, len(ts_list)-1):
        previous_end = ts.end_time()
        ts = ts[:-1].append(ts_list[i][previous_end:])
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


def train_models(models:list, ts_train_list_piped, ts_train_weather_list_piped=None):
    '''This function trains a list of models on the training data'''
    for model in models:
        model.fit(ts_train_list_piped, future_covariates=ts_train_weather_list_piped)
    return models

def make_sklearn_models(list_sklearn_models, encoders, N_LAGS, N_AHEAD, LIKLIHOOD):
    model_instances = []
    for model in list_sklearn_models:
        model = model(lags=N_LAGS,
                      lags_future_covariates=[0],
                      add_encoders=encoders, 
                      output_chunk_length=N_AHEAD, 
                      likelihood=LIKLIHOOD)
        model_instances.append(model)
    return model_instances


def calc_error_scores(metrics, ts_predictions_inverse, trg_inversed):
    metrics_scores = {}
    for metric in metrics:
        score = metric(ts_predictions_inverse, trg_inversed)
        metrics_scores[metric.__name__] = score
    return metrics_scores

def get_error_metric_table(metrics, ts_predictions_per_model, trg_test_inversed):

    error_metric_table = {}
    for model_name, ts_predictions_inverse in ts_predictions_per_model.items():
        ts_predictions_inverse, trg_inversed = make_index_same(ts_predictions_inverse, trg_test_inversed)
        metrics_scores = calc_error_scores(metrics, ts_predictions_inverse, trg_inversed)
        error_metric_table[model_name] = metrics_scores
    
    df_metrics  = pd.DataFrame(error_metric_table).T
    return df_metrics
