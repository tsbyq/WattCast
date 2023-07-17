# %%
import torch
import os
from torch.utils.data import Dataset, DataLoader
from darts import TimeSeries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler, RobustScaler
from darts.dataprocessing.transformers.boxcox import BoxCox
from darts.dataprocessing.transformers.scaler import Scaler
from darts.dataprocessing import Pipeline

from darts.metrics import mape, mse, rmse
import time
import json
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from darts.models import (
    BlockRNNModel, NBEATSModel, RandomForest, 
    LightGBMModel, XGBModel, LinearRegressionModel, TFTModel, TransformerModel
)

import wandb


MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), 'models')
if not os.path.exists(MODEL_DIR): 
    os.makedirs(MODEL_DIR)
    print(f"Created directory: {MODEL_DIR}")
BEST_SCORE = 99999


# %%
class Config:
    '''
    Class to store config parameters, to circumvent the wandb.config when combining multiple models.
    '''

    def __init__(self):
        self.data = {}

    def __getattr__(self, key):
        if key in self.data:
            return self.data[key]
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == 'data':
            # Allow normal assignment for the 'data' attribute
            super().__setattr__(key, value)
        else:
            self.data[key] = value

    def __delattr__(self, key):
        if key in self.data:
            del self.data[key]
        else:
            raise AttributeError(f"'Config' object has no attribute '{key}'")

    def __len__(self):
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    @classmethod
    def from_dict(cls, data):
        config = cls()
        for key, value in data.items():
            config[key] = value  # Preserve nested dictionaries without converting
        return config


def prepare_timeseries(
        df,
        datetime_col='datetime',
        tgt_col='Electric_MW',
        cov_cols=['Temperature'],
        num_chunks=4,
        split_pcts=[0.7, 0.2, 0.1],
        config=None,
        pipe=True
):
    # Assuming your DataFrame is named 'df' with columns 'datetime' and 'value'
    tgt_series = TimeSeries.from_dataframe(df, datetime_col, tgt_col)
    cov_series = TimeSeries.from_dataframe(df, datetime_col, cov_cols)

    assert abs(sum(split_pcts)-1.0) < 1e-6, "Split percentages must sum to 1.0"
    assert len(tgt_series) == len(cov_series), "Target and covariate series must be the same length"

    # Define the seasonal chunks
    chunk_length = len(tgt_series) // num_chunks

    # Initialize lists to store the split datasets
    train_tgt_timeseriess = []
    val_tgt_timeseriess = []
    test_tgt_timeseriess = []

    train_cov_timeseriess = []
    val_cov_timeseriess = []
    test_cov_timeseriess = []

    # Split the dataset into chunks and apply split percentages
    for i in range(num_chunks):
        start_index = i * chunk_length
        end_index = (i + 1) * chunk_length

        tgt_chunk = tgt_series[start_index:end_index]
        cov_chunk = cov_series[start_index:end_index]

        split_sizes = np.array(split_pcts) * len(tgt_chunk)

        train_tgt_ts = tgt_chunk[:int(split_sizes[0])]
        val_tgt_ts = tgt_chunk[int(split_sizes[0]):int(split_sizes[0] + split_sizes[1])]
        test_tgt_ts = tgt_chunk[int(split_sizes[0] + split_sizes[1]):]

        train_cov_ts = cov_chunk[:int(split_sizes[0])]
        val_cov_ts = cov_chunk[int(split_sizes[0]):int(split_sizes[0] + split_sizes[1])]
        test_cov_ts = cov_chunk[int(split_sizes[0] + split_sizes[1]):]

        train_tgt_timeseriess.append(train_tgt_ts)
        val_tgt_timeseriess.append(val_tgt_ts)
        test_tgt_timeseriess.append(test_tgt_ts)

        train_cov_timeseriess.append(train_cov_ts)
        val_cov_timeseriess.append(val_cov_ts)
        test_cov_timeseriess.append(test_cov_ts)

    out = {
        'train_tgt_timeseriess': train_tgt_timeseriess,
        'val_tgt_timeseriess': val_tgt_timeseriess,
        'test_tgt_timeseriess': test_tgt_timeseriess,
        'train_cov_timeseriess': train_cov_timeseriess,
        'val_cov_timeseriess': val_cov_timeseriess,
        'test_cov_timeseriess': test_cov_timeseriess,
    }

    if pipe and config is not None:
        out = {**out, **pipe_timeseriess(out, config)}

    return out

def pipe_timeseriess(dict_timeseriess, config):

    # Preprocessing Pipeline
    pipeline_tgt = Pipeline([
        BoxCox() if config.boxcox else Scaler(MinMaxScaler()), # double scale in case boxcox is turned off
        Scaler(MinMaxScaler()),
    ])

    train_tgt_timeseriss_piped = pipeline_tgt.fit_transform(dict_timeseriess['train_tgt_timeseriess'])
    val_tgt_timeseriss_piped = pipeline_tgt.transform(dict_timeseriess['val_tgt_timeseriess'])
    test_tgt_timeseriss_piped = pipeline_tgt.transform(dict_timeseriess['test_tgt_timeseriess'])

    # Weather Pipeline
    pipeline_cov = Pipeline([Scaler(RobustScaler())])
    train_cov_timeseriess = pipeline_cov.fit_transform(dict_timeseriess['train_cov_timeseriess'])
    val_cov_timeseriess = pipeline_cov.fit_transform(dict_timeseriess['val_cov_timeseriess'])
    test_cov_timeseriess = pipeline_cov.fit_transform(dict_timeseriess['test_cov_timeseriess'])

    out = {
        'train_tgt_timeseriess_piped': train_tgt_timeseriss_piped,
        'val_tgt_timeseriess_piped': val_tgt_timeseriss_piped,
        'test_tgt_timeseriess_piped': test_tgt_timeseriss_piped,
        'train_cov_timeseriess_piped': train_cov_timeseriess,
        'val_cov_timeseriess_piped': val_cov_timeseriess,
        'test_cov_timeseriess_piped': test_cov_timeseriess,
        'pipe_tgt': pipeline_tgt,
        'pipe_cov': pipeline_cov,
    }

    return out

def prepare_config(config):
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
    # config.eval_stride = 1
    # config.model_path = os.path.join(MODEL_DIR, config.model_name)

    return config


def init_model(config):

    '''Returns model instance, based on the config.'''

    model = config.model

    # for torch models
    optimizer_kwargs = {}
    try:
        optimizer_kwargs['lr'] = config.lr
    except:
        optimizer_kwargs['lr'] = 1e-3
    
    pl_trainer_kwargs = {
        'max_epochs': 20,
        'accelerator': 'gpu',
        'devices': [0],
        # 'callbacks': [EarlyStopping(monitor='val_loss', patience=5, mode='min')],
        #'logger': WandbLogger(log_model='all'),
    }

    # schedule_kwargs = {
    #     'patience': 2,
    #     'factor': 0.5,
    #     'min_lr': 1e-5,
    #     'verbose': True
    # }
    schedule_kwargs = {
        'T_max': 30,
        'eta_min': 1e-5,
        'verbose': True
    }


    if model == 'xgb':
        try:
            xgb_kwargs = {
                'n_estimators': config.n_estimators,
                'max_depth': config.max_depth,
                'learning_rate': config.learning_rate,
                'min_child_weight': config.min_child_weight,
                'objective': config.objective,
                'reg_lambda': config.reg_lambda,
                'early_stopping_rounds': 5
            }
        except:
            xgb_kwargs ={}

        model = XGBModel(
            lags=config.n_lags,
            lags_past_covariates=config.n_lags if config.use_past_cov else None,
            lags_future_covariates=[0] if config.use_future_cov else None,
            add_encoders=config.datetime_encoders, 
            output_chunk_length=config.n_ahead, 
            likelihood=config.liklihood,
            random_state=42,
            **xgb_kwargs
        )
    
    elif model == 'lgbm':
        try:
            lightgbm_kwargs = {
                'n_estimators': config.n_estimators,
                'max_depth': config.max_depth,
                'learning_rate': config.learning_rate,
                'min_child_weight': config.min_child_weight,
                'num_leaves': config.num_leaves,
                'objective': config.objective,
                'min_child_samples': config.min_child_samples
            }

        except:
            lightgbm_kwargs = {}
        
        model = LightGBMModel(
            lags=config.n_lags,
            lags_past_covariates=config.n_lags if config.use_past_cov else None,
            lags_future_covariates=[0] if config.use_future_cov else None,
            add_encoders=config.datetime_encoders,
            output_chunk_length=config.n_ahead,
            likelihood=config.liklihood,
            random_state=42,
            **lightgbm_kwargs
        )

    elif model == 'rf':
        rf_kwargs = {
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
            'min_samples_split': config.min_samples_split,
            'min_samples_leaf': config.min_samples_leaf,
        }

        model = RandomForest(
            lags=config.n_lags,
            lags_past_covariates=config.n_lags if config.use_past_cov else None,
            lags_future_covariates=[0] if config.use_future_cov else None,
            add_encoders=config.datetime_encoders,
            output_chunk_length=config.n_ahead,
            random_state=42,
            **rf_kwargs
        )


    elif model == 'nbeats':
        try:
            nbeats_kwargs = {
                'batch_size': config.batch_size,
            }

        except:
            nbeats_kwargs = {} 

        model = NBEATSModel(
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            # lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_cls=CosineAnnealingLR,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
            **nbeats_kwargs
        )

    
    elif model == 'gru':
        try:
            rnn_kwargs = {
                'hidden_dim': config.hidden_dim,
                'n_rnn_layers': config.n_rnn_layers,
                'batch_size': config.batch_size,
                'dropout': config.dropout,
            }
        except:
            rnn_kwargs = {}

        model = BlockRNNModel(  
            model = 'GRU',
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            # lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_cls=CosineAnnealingLR,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
            **rnn_kwargs
        )
        

    elif model == 'tft':

        try:
            transformer_kwargs = {
                'hidden_size': config.hidden_dim,
                'lstm_layers': config.n_rnn_layers,
                'batch_size': config.batch_size,
                'dropout': config.dropout,
                'num_attention_heads': config.num_attention_heads,
            }
        except:
            transformer_kwargs = {}

        
        model = TFTModel(
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            # lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_cls=CosineAnnealingLR,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
            **transformer_kwargs
        )
        
    elif model == 'transformer':

        try:
            transformer_kwargs = {
                'd_model': config.d_model,
                'nhead': config.nhead,
                'num_encoder_layers': config.num_encoder_layers,
                'num_decoder_layers': config.num_decoder_layers,
                'batch_size': config.batch_size,
                'dropout': config.dropout,
                'num_attention_heads': config.num_attention_heads,
            }
        except:
            transformer_kwargs = {}

        model = TransformerModel(
            input_chunk_length=config.n_lags,
            output_chunk_length=config.n_ahead,
            add_encoders=config.datetime_encoders,
            likelihood=config.liklihood,
            pl_trainer_kwargs=pl_trainer_kwargs,
            optimizer_kwargs=optimizer_kwargs,
            # lr_scheduler_cls=ReduceLROnPlateau,
            lr_scheduler_cls=CosineAnnealingLR,
            lr_scheduler_kwargs=schedule_kwargs,
            random_state=42,
        )
    return model


def train_loop(model, dict_tss, config):
    print("Training model...")
    start = time.time()

    print(f"==>> config.use_past_cov: {config.use_past_cov}")
    print(f"==>> config.use_future_cov: {config.use_future_cov}")
    use_past_cov = model.supports_past_covariates and bool(config.use_past_cov)
    use_future_cov = model.supports_future_covariates and bool(config.use_future_cov)
    print(f"==>> use_future_cov: {use_future_cov}")
    print(f"==>> use_past_cov: {use_past_cov}")

    try:
        model.fit(
            series=dict_tss['train_tgt_timeseriess_piped'], 
            past_covariates=dict_tss['train_cov_timeseriess_piped'] if use_past_cov else None, 
            future_covariates=dict_tss['train_cov_timeseriess_piped'] if use_future_cov else None,
            val_series=dict_tss['train_tgt_timeseriess_piped'], 
            val_past_covariates=dict_tss['train_cov_timeseriess_piped'],
            verbose=True,
        )
    except:
        try:
            model.fit(
                series=dict_tss['train_tgt_timeseriess_piped'], 
                past_covariates=dict_tss['train_cov_timeseriess_piped'] if use_past_cov else None, 
                future_covariates=dict_tss['train_cov_timeseriess_piped'] if use_future_cov else None,
                verbose=True,
            )
        except:
            model.fit(
                series=dict_tss['train_tgt_timeseriess_piped'], 
                past_covariates=dict_tss['train_cov_timeseriess_piped'] if use_past_cov else None, 
                future_covariates=dict_tss['train_cov_timeseriess_piped'] if use_future_cov else None,
            ) 

    # # Calculate validation loss using backtest
    # backtest_results = model.backtest(
    #     series=dict_tss['val_tgt_timeseriess_piped'], 
    #     past_covariates=dict_tss['val_cov_timeseriess_piped'],
    #     stride=config.eval_stride,
    #     reduction=np.mean,
    #     retrain=False,
    #     verbose=False,
    #     metric=rmse,  # Choose the desired metric
    # )
    # val_loss = np.mean(backtest_results)

    end = time.time()
    runtime = end - start
    return model, runtime


def eval_plot(dict_tss, historics, config):
    f, axs = plt.subplots(len(historics), 1, figsize=(25, 2*len(historics)))
    pipe = dict_tss['pipe_tgt']
    for i, (val_pred, val_gt) in enumerate(zip(historics, dict_tss['val_tgt_timeseriess_piped'])):
        for j, seq in tqdm(enumerate(val_pred), total=len(val_pred), desc=f'Plotting prediction for chunk {i+1}'):
            axs[i].plot(
                pipe.inverse_transform(seq).pd_series().index,
                pipe.inverse_transform(seq).pd_series().values,
                lw=0.6
            )
        axs[i].plot(
            pipe.inverse_transform(val_gt).pd_series().index,
            pipe.inverse_transform(val_gt).pd_series().values,
            lw=1, color='black'
        )
        axs[i].set_title(f'Chunk {i+1}: {val_gt.start_time().strftime("%Y-%m-%d %H:%M")} to {val_gt.end_time().strftime("%Y-%m-%d %H:%M")}')
        axs[i].set_ylabel('Power Demand (MW)')
        axs[i].set_ylim(0, )
        axs[i].axhline(0, color='black', lw=2)
    f.tight_layout()
    f.suptitle(f'Validation Results', fontsize=16, y=1.02)
    return f


def eval_loop(model, dict_tss, config):
    print("Evaluating model...")
    historics = model.historical_forecasts(
        series=dict_tss['val_tgt_timeseriess_piped'], 
        past_covariates=dict_tss['val_cov_timeseriess_piped'] if config.use_past_cov else None,
        future_covariates= dict_tss['val_cov_timeseriess_piped'] if config.use_future_cov else None,
        # start=ts.get_index_at_point(n_lags),
        verbose=False,
        stride=config.eval_stride, 
        forecast_horizon=config.n_ahead, 
        retrain=False, 
        last_points_only=False,
    )

    test_out = model.backtest(
        series=dict_tss['val_tgt_timeseriess_piped'], 
        # past_covariates=dict_tss['val_cov_timeseriess_piped'],
        historical_forecasts=historics,
        stride=config.eval_stride,
        reduction=np.mean,
        retrain=False,
        verbose=False,
        metric=rmse,
    )

    metric = np.mean(test_out)
    fig_compare = eval_plot(dict_tss, historics, config)

    return metric, fig_compare


def train_eval():

    wandb.init(project=WB_PROJECT, entity="wattcast")
    # wandb.config.update(config_run)
    config = wandb.config
    print(f"==>> config: {config}")
    config.update(config_run)
    config = prepare_config(config)
    print(f"==>> config: {config}")
    wandb.config.update(config_run)
    
    print("Getting data...")
    df = pd.read_csv(config.data_path, parse_dates=['datetime'])
    dict_tss = prepare_timeseries(df, config=config, pipe=True)
    model = init_model(config)
    model, runtime = train_loop(model, dict_tss, config)
    metric, fig_compare = eval_loop(model, dict_tss, config)
    # model.early_stopping_kwargs = {'patience': 5, 'min_delta': 0, 'threshold': val_loss}
    # if metric < BEST_SCORE:
    #     print(f"Current best model: {config.model}-{config.spatial_scale}_{config.location} achieved score={metric}")
    #     model.save(os.path.join(MODEL_DIR, f"{config.model}-{config.spatial_scale}_{config.location}.pt"))
    #     BEST_SCORE = metric

    wandb.log({'test_score': metric})
    wandb.log({'runtime': runtime})
    wandb.log({'fig_compare': wandb.Image(fig_compare)})

    wandb.finish()


########################################################################################
# Sweep for hyperparameter tuning
########################################################################################
if __name__ == '__main__':
    wandb.login()

    # WB_PROJECT = 'East_Portland_tuning'
    WB_PROJECT = 'East_Portland_tuning_2'
    
    sweeps = 30
    models = [
        'rf',
        'xgb', 
        'gru', 
        'lgbm',  
        'nbeats',
        # 'tft'
    ]

    for model in models:
        # place holder initialization of config file (will be updated in train_eval_light()
        config_run = {
            'spatial_scale': 'City',
            'temp_resolution': 60,
            'location': 'East Portland',
            'model': model,
            'horizon_in_hours': 4,
            'lookback_in_hours': 48,
            'boxcox': True,
            'liklihood': None,
            'weather': True,
            'holiday': True,
            'datetime_encodings': False,
            # 'data_path': 'E:/GitHub/Forked_Repos/WattCast/data/clean_data/Portland.csv',
            'data_path': 'F:/Han/OE_HW_Portland_WC/WattCast/data/clean_data/Portland.csv'
        }
        
        with open(f'sweep_configurations/config_sweep_{model}.json', 'r') as fp:
            sweep_config = json.load(fp)                  

        sweep_config['name'] = model + 'sweep' + config_run['spatial_scale'] + '_' + config_run['location'] + '_' + str(config_run['temp_resolution']) + '_horizon=' + str(config_run['horizon_in_hours']) + '_lookback=' + str(config_run['lookback_in_hours'])
        print(f"==>> sweep_config: {sweep_config}")
        sweep_id = wandb.sweep(sweep_config, project=WB_PROJECT, entity="wattcast")
        wandb.agent(sweep_id, train_eval, count=sweeps)





