# train_eval.py
import os
import wandb
import pandas as pd
import numpy as np
import darts
from darts import TimeSeries
from darts.utils.missing_values import extract_subseries
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from utils import review_subseries, get_longest_subseries_idx, train_models, predict_testset

dir_path = os.path.join(os.getcwd(), 'data', 'clean_data')
model_dir = os.path.join(os.getcwd(), 'models')


def data_pipeline(config):

    if config.temp_resolution == 60:
        timestep_encoding = ["hour"] 
    elif config.temp_resolution == 15:
        timestep_encoding = ['quarter']
    else:
        timestep_encoding = ["hour", "minute"]


    datetime_encoders =  {
                        "cyclic": {"future": timestep_encoding}, 
                        "position": {"future": ["relative",]},
                        "datetime_attribute": {"future": ["dayofweek", "week"]},
                        'position': {'past': ['relative'], 'future': ['relative']},
                }

    datetime_encoders = datetime_encoders if config.datetime_encodings else None

    config['datetime_encoders'] = datetime_encoders


    config.timesteps_per_hour = int(60 / config.temp_resolution)
    config.n_lags = config.lookback_in_hours * config.timesteps_per_hour
    config.n_ahead = config.horizon_in_hours * config.timesteps_per_hour
    config.eval_stride = int(np.sqrt(config.n_ahead)) # evaluation stride, how often to evaluate the model, in this case we evaluate every n_ahead steps

    # Loading Data
    df_train = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/train_target')
    df_val = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/val_target')
    df_test = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'),key=f'{config.location}/{config.temp_resolution}min/test_target')

    df_cov_train = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/train_cov')
    df_cov_val = pd.read_hdf(os.path.join(dir_path, f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/val_cov')
    df_cov_test = pd.read_hdf(os.path.join(dir_path,f'{config.spatial_scale}.h5'), key=f'{config.location}/{config.temp_resolution}min/test_cov')

    # into darts format
    ts_train = darts.TimeSeries.from_dataframe(df_train, freq=str(config.temp_resolution) + 'min')
    ts_train = extract_subseries(ts_train)
    ts_val = darts.TimeSeries.from_dataframe(df_val, freq=str(config.temp_resolution) + 'min')
    ts_val = extract_subseries(ts_val)
    ts_test = darts.TimeSeries.from_dataframe(df_test, freq=str(config.temp_resolution) + 'min')
    ts_test = extract_subseries(ts_test)

    # Covariates
    if config.weather:
        ts_cov_train = darts.TimeSeries.from_dataframe(df_cov_train, freq=str(config.temp_resolution) + 'min')
        ts_cov_val = darts.TimeSeries.from_dataframe(df_cov_val, freq=str(config.temp_resolution) + 'min')
        ts_cov_test = darts.TimeSeries.from_dataframe(df_cov_test, freq=str(config.temp_resolution) + 'min')
    else:
        ts_cov_train = None
        ts_cov_val = None
        ts_cov_test = None

    # Reviewing subseries to make sure they are long enough
    ts_train, ts_cov_train = review_subseries(ts_train, config.n_lags + config.n_ahead, ts_cov_train)
    ts_val, ts_cov_val = review_subseries(ts_val, config.n_lags + config.n_ahead, ts_cov_val)
    ts_test, ts_cov_test = review_subseries(ts_test, config.n_lags +config.n_ahead, ts_cov_test)

    # getting the index of the longest subseries, to be used for evaluation later
    config.longest_ts_val_idx = get_longest_subseries_idx(ts_val)
    config.longest_ts_test_idx = get_longest_subseries_idx(ts_test)

    # Preprocessing Pipeline
    pipeline = Pipeline( # missing values have been filled in the 'data_prep.ipynb'
                    [
                    BoxCox() if config.boxcox else None,
                    Scaler(MinMaxScaler()),
                    ]
                    )
    ts_train_piped = pipeline.fit_transform(ts_train)
    ts_val_piped = pipeline.transform(ts_val)
    ts_test_piped = pipeline.transform(ts_test)

    # Weather Pipeline
    if config.weather:
        pipeline_weather = Pipeline([Scaler(RobustScaler())])
        ts_train_weather_piped = pipeline_weather.fit_transform(ts_cov_train)
        ts_val_weather_piped = pipeline_weather.transform(ts_cov_val)
        ts_test_weather_piped = pipeline_weather.transform(ts_cov_test)
    else:
        ts_train_weather_piped = None
        ts_val_weather_piped = None
        ts_test_weather_piped = None

    trg_train_inversed = pipeline.inverse_transform(ts_train_piped, partial=True) 
    trg_val_inversed = pipeline.inverse_transform(ts_val_piped, partial=True)[config.longest_ts_val_idx] 
    trg_test_inversed = pipeline.inverse_transform(ts_test_piped, partial=True)[config.longest_ts_test_idx]

    return pipeline, ts_train_piped, ts_val_piped, ts_test_piped, ts_train_weather_piped, ts_val_weather_piped, ts_test_weather_piped, trg_train_inversed, trg_val_inversed, trg_test_inversed



def get_model_instance(config):

    model = config.model

    if model == 'xgb':

        try:
            xgb_kwargs = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'learning_rate': config.learning_rate,
            'min_child_weight': config.min_child_weight,
            'objective': config.objective,
            'reg_lambda': config.reg_lambda,
            'early_stopping_rounds': 10
            }
        except:
            xgb_kwargs ={}

        model = XGBModel(lags=config.n_lags,
                            lags_future_covariates=[0],
                            add_encoders=config.datetime_encoders, 
                            output_chunk_length=config.n_ahead, 
                            likelihood=config.liklihood,
                            random_state=42,
                            **xgb_kwargs
                            )
    
    elif model == 'gru':

        optimizer_kwargs = {}
        optimizer_kwargs['lr'] = config.lr
        
        pl_trainer_kwargs = {
        'max_epochs': 50,
        'accelerator': 'gpu',
        'devices': [0],
        'callbacks': [EarlyStopping(monitor='val_loss', patience=5, mode='min')],
        'logger': WandbLogger(log_model='all'),
    }

        schedule_kwargs = {
            'patience': 2,
            'factor': 0.5,
            'min_lr': 1e-5,
            'verbose': True
            }

        model = RNNModel(  
                        model = 'GRU',
                        input_chunk_length=config.n_lags,
                        output_chunk_length=config.n_ahead,
                        hidden_dim=config.hidden_dim,
                        n_rnn_layers=config.n_rnn_layers,
                        batch_size=config.batch_size,
                        dropout=config.dropout,
                        add_encoders=config.datetime_encoders,
                        likelihood=None,
                        pl_trainer_kwargs=pl_trainer_kwargs,
                        optimizer_kwargs=optimizer_kwargs,
                        lr_scheduler_cls=ReduceLROnPlateau,
                        lr_scheduler_kwargs=schedule_kwargs,
                        random_state=42,
                    )

    return [model]
