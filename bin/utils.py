# utils.py

'''This file contains utility functions that are used in the notebooks.'''

import os
import pandas as pd
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import holidays
from scipy.stats import boxcox
from scipy.signal import find_peaks_cwt
from fastdtw import fastdtw
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer


### Pytorch and Darts Helper Functions  

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


def calc_metrics(df_compare, metrics):
    "calculates metrics for a dataframe with a ground truth column and predictions, ground truth column must be the first column"
    metric_series_list = {}
    for metric in metrics:
        metric_name = metric.__name__
        metric_result = df_compare.apply(lambda x: metric(x, df_compare.iloc[:,0]), axis=0)
        if metric.__name__ == 'mean_squared_error':
            metric_result = np.sqrt(metric_result)
            metric_name = 'root_mean_squared_error'
        elif metric.__name__ == 'r2_score':
            metric_result = 1 - metric_result

        metric_series_list[metric_name] = metric_result

    df_metrics = pd.DataFrame(metric_series_list).iloc[1:,:]
    return df_metrics


### Feature Engineering


def timeseries_peak_feature_extractor(df):
    'Extracts peak count, maximum peak height, and time of two largest peaks for each day in a pandas dataframe time series'
    
    # Find peaks
    peak_idx = find_peaks_cwt(df.values.flatten(), widths=3, max_distances=[48], window_size=96)
    
    # Convert peak indices to datetime indices
    peak_times = [df.index[i] for i in peak_idx]
    
    # Group peaks by day
    peak_days = pd.Series(peak_times).dt.date
    
    # Count peaks for each day
    daily_peak_count = peak_days.value_counts().sort_index()
    
    # Find maximum and second maximum peak height and time for each day
    daily_peak_height = []
    daily_peak_time = []
    daily_second_peak_height = []
    daily_second_peak_time = []
    
    for day in daily_peak_count.index:
        day_peaks = [peak_idx[i] for i in range(len(peak_idx)) if peak_times[i].date() == day]
        day_peak_vals = [df.values[i] for i in day_peaks]
        
        max_idx = np.argmax(day_peak_vals)
        daily_peak_height.append(day_peak_vals[max_idx][0])
        daily_peak_time.append((day_peaks[max_idx] % 96))
        
        if len(day_peak_vals) > 1:
            day_peak_vals[max_idx] = -np.inf
            second_max_idx = np.argmax(day_peak_vals)
            daily_second_peak_height.append(day_peak_vals[second_max_idx][0])
            daily_second_peak_time.append((day_peaks[second_max_idx] % 96))
        else:
            daily_second_peak_height.append(0)
            daily_second_peak_time.append(0)
    
    # Combine results into output DataFrame
    output_df = pd.DataFrame({'peak_count': daily_peak_count.values,
                              'height_highest_peak': daily_peak_height,
                              'time_highest_peak': daily_peak_time,
                              'height_second_highest_peak': daily_second_peak_height,
                              'time_second_highest_peak': daily_second_peak_time},
                             index=daily_peak_count.index)
    
    return output_df




def calc_rolling_sum_of_load(df, n_days):
    df['rolling_sum'] = df.sum(axis=1).rolling(n_days).sum().shift(1)
    df = df.dropna()
    return df


def create_datetime_features(df):
    df['day_of_week'] = df.index.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    df.drop('day_of_week', axis=1, inplace=True)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    # is weekend
    df['is_weekend'] = df.index.dayofweek > 4
    df.drop('month', axis=1, inplace=True)
    return df

def create_holiday_features(df, df_holidays, df_holiday_periods=None):

    df_1 = days_until_next_holiday_encoder(df, df_holidays)
    df_2 = days_since_last_holiday_encoder(df, df_holidays)

    df_3 = pd.concat([df_1, df_2], axis=1)

    if df_holiday_periods is not None:
        df_3 = pd.concat([df_3, df_holiday_periods], axis=1)

    df_3 = df_3.loc[~df_3.index.duplicated(keep='first')]

    df_3 = df_3.reindex(df.index, fill_value=0)

    return df_3


def days_until_next_holiday_encoder(df, df_holidays):

    df_concat = pd.concat([df, df_holidays], axis=1)
    df_concat["days_until_next_holiday"] = 0
    for ind in df_concat.index:
        try:
            next_holiday = df_concat["holiday_dummy"].loc[ind:].first_valid_index()
            days_until_next_holiday = (next_holiday - ind).days
            df_concat.loc[ind, "days_until_next_holiday"] = days_until_next_holiday
        except:
            pass

    return df_concat[["days_until_next_holiday"]]


def days_since_last_holiday_encoder(df, df_holidays):

    df_concat = pd.concat([df, df_holidays], axis=1)
    df_concat["days_since_last_holiday"] = 0
    for ind in df_concat.index:
        next_holiday = df_concat["holiday_dummy"].loc[:ind].last_valid_index()
        days_since_last_holiday = (ind - next_holiday).days
        df_concat.loc[ind, "days_since_last_holiday"] = days_since_last_holiday

    return df_concat[["days_since_last_holiday"]]


def get_year_list(df):
    'Return the list of years in the historic data'
    years = df.index.year.unique()
    years = years.sort_values()
    return list(years)


def get_holidays(years, shortcut):

    country = getattr(holidays, shortcut)
    holidays_dict = country(years=years)
    df_holidays = pd.DataFrame(holidays_dict.values(), index=holidays_dict.keys())
    df_holidays[0] = 1
    df_holidays_dummies = df_holidays
    df_holidays_dummies.columns = ["holiday_dummy"]
    df_holidays_dummies.index = pd.DatetimeIndex(df_holidays.index)
    df_holidays_dummies = df_holidays_dummies.sort_index()

    return df_holidays_dummies



### Transformations & Cleaning


def standardize_format(df, timestep, location, unit:str):
    df = df.sort_index()
    df = remove_duplicate_index(df)
    df = df.resample(f'{timestep}T').mean()
    df.index.name = 'datetime'
    df.columns = [f'load_{location}_{unit}']
    return df

# a function to do a train test split, the train should be a full year and the test should be a tuple of datasets, each one month long


def save_train_val_test_datasets(df, train_start, train_end, val_start, val_end, test_start, test_end, spatial_scale):

    train = df.loc[train_start:train_end]
    val = df.loc[val_start:val_end]
    test = df.loc[test_start:test_end]
    key = train.columns[0].split('_')[1]
    # Save the dataframes
    train.to_csv(f"data/cleaned_data/{spatial_scale}/power/{key}_train.csv")
    val.to_csv(f"data/cleaned_data/{spatial_scale}/power/{key}_val.csv")
    test.to_csv(f"data/cleaned_data/{spatial_scale}/power/{key}_test.csv")


def remove_non_positive_values(df):
    'Removes all non-positive values from a dataframe and interpolates the missing values'
    df[df<=0] = np.nan
    df = df.interpolate(method='linear', axis=0).ffill().bfill()
    df.dropna(inplace=True)
    return df





def remove_days(df_raw, p=0.05):
    'Removes days with less than 5% of average total energy consumption of all days'
    df = df_raw.copy()
    days_to_remove = []
    days = list(set(df.index.date))
    threshold = df.groupby(df.index.date).sum().quantile(p).values[0]
    for day in days:
        if df.loc[df.index.date == day].sum().squeeze() < threshold:
            days_to_remove.append(day)

    mask = np.in1d(df.index.date, days_to_remove)

    df = df[~mask].dropna()

    print(f'Removed {len(days_to_remove)} days with less than {p*100}% of average total energy consumption of all days')
    
    return df


def remove_duplicate_index(df):
    df = df.loc[~df.index.duplicated(keep='first')]
    return df

def timeseries_dataframe_pivot(df):
    df_ = df.copy()
    df_['date'] = df_.index.date
    df_['time'] = df_.index.time

    df_pivot = df_.pivot(index='date', columns='time')

    n_days, n_timesteps = df_pivot.shape

    df_pivot.dropna(thresh = n_timesteps // 5, inplace=True)

    df_pivot = df_pivot.fillna(method='ffill', axis = 0)

    df_pivot = df_pivot.droplevel(0, axis=1)

    df_pivot.columns.name = None

    df_pivot.index = pd.DatetimeIndex(df_pivot.index)

    return df_pivot


def unpivot_timeseries_dataframe(df: pd.DataFrame, column_name: str = "Q"):

    df_unstack = df.T.unstack().to_frame().reset_index()
    df_unstack.columns = ["date", "time", "{}".format(column_name)]
    df_unstack["date_str"] = df_unstack["date"].apply(
        lambda t: datetime.datetime.strftime(t, format="%Y-%m-%d")
    )
    df_unstack["time_str"] = df_unstack["time"].apply(
        lambda t: " {}:{}:{}".format(t.hour, t.minute, t.second)
    )
    df_unstack["datetime_str"] = df_unstack["date_str"] + df_unstack["time_str"]
    df_unstack = df_unstack.set_index(
        pd.to_datetime(df_unstack["datetime_str"], format="%Y-%m-%d %H:%M:%S")
    )[[column_name]]
    df_unstack.index.name = "datetime"

    return df_unstack


def boxcox_transform(dataframe, lam = None):
    """
    Perform a Box-Cox transform on a pandas dataframe timeseries.
    
    Args:
    dataframe (pandas.DataFrame): Pandas dataframe containing the timeseries to transform.
    lam (float): The lambda value to use for the Box-Cox transformation.
    
    Returns:
    transformed_dataframe (pandas.DataFrame): Pandas dataframe containing the transformed timeseries.
    """
    transformed_dataframe = dataframe.copy()
    for column in transformed_dataframe.columns:
        transformed_dataframe[column], lam = boxcox(transformed_dataframe[column], lam)
    return transformed_dataframe, lam


def inverse_boxcox_transform(dataframe, lam):
    """
    Inverse the Box-Cox transform on a pandas dataframe timeseries.
    
    Args:
    dataframe (pandas.DataFrame): Pandas dataframe containing the timeseries to transform.
    lam (float): The lambda value used for the original Box-Cox transformation.
    
    Returns:
    transformed_dataframe (pandas.DataFrame): Pandas dataframe containing the inverse-transformed timeseries.
    """
    transformed_dataframe = dataframe.copy()
    for column in transformed_dataframe.columns:
        if lam == 0:
            transformed_dataframe[column] = np.exp(transformed_dataframe[column])
        else:
            transformed_dataframe[column] = np.exp(np.log(lam * transformed_dataframe[column] + 1) / lam)
    return transformed_dataframe


def post_process_xgb_predictions(predictions, boxcox_bool, scaler=None, lam = None):
    'Post-process the predictions of the Multi-Output XGBoost model'
    predictions_reshaped = predictions.reshape(-1,1).flatten()
    # set negative predictions to 5th percentile of the training data
    predictions_reshaped[predictions_reshaped < 0] = np.quantile(predictions_reshaped, 0.05)   
    # reverse the scaling and boxcox transformation of the predictions
    if scaler is not None:
        predictions_reshaped = scaler.inverse_transform(predictions_reshaped.reshape(-1,1)).flatten()
    if boxcox_bool:
        predictions_reshaped = inverse_boxcox_transform(pd.DataFrame(predictions_reshaped), lam).values.flatten()
    return predictions_reshaped


# model evaluation

def dtw_metric(y_true, y_pred):
    distance, path = fastdtw(y_true, y_pred)
    return distance

dtw_scorer = make_scorer(dtw_metric, greater_is_better=False)

rmse_scorer = make_scorer(mean_squared_error, greater_is_better=False)