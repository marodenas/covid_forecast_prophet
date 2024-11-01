import time
import optuna
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK,space_eval
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import pandas as pd
from typing import Any, Dict
from sklearn.model_selection import ParameterGrid
import malib.models.split as msp
from itertools import product
from tqdm import tqdm

# Función para calcular las métricas
def calculate_metrics(df_cv):
    df_p = performance_metrics(df_cv)
    rmse = df_p['rmse'].mean()
    mae = df_p['mae'].mean()
    mape = df_p['mape'].mean()
    rmse_mae = rmse + mae
    return rmse, mae, mape, rmse_mae

# Función objetivo para Optuna
def run_optuna(train_df, cutoff, n_trials=250):
    def objective(trial):
        params = {
            'n_changepoints': trial.suggest_int('n_changepoints', 20, 100),
            'changepoint_range': trial.suggest_float('changepoint_range', 0.7, 0.9),
            'seasonality_mode': trial.suggest_categorical('seasonality_mode', ['additive', 'multiplicative']),
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.01, 0.5),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10.0),
            'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 100.0),
            'interval_width': trial.suggest_float('interval_width', 0.8, 0.95)
        }
        m = Prophet(**params)
        m.fit(train_df)
        df_cv = cross_validation(m, initial=cutoff['initial'], period=cutoff['period'], horizon=cutoff['horizon'])
        rmse, mae, mape, rmse_mae = calculate_metrics(df_cv)
        trial.set_user_attr('other_metrics', (rmse, mae, mape, rmse_mae))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial), n_trials=n_trials)
    best_params = study.best_trial.params
    best_metrics = study.best_trial.user_attrs['other_metrics']
    return best_metrics, best_params

# Función objetivo para Hyperopt
def objective_hyperopt(params, train_df, cutoff):
    params['n_changepoints'] = int(params['n_changepoints'])
    m = Prophet(**params)
    m.fit(train_df)
    df_cv = cross_validation(m, initial=cutoff['initial'], period=cutoff['period'], horizon=cutoff['horizon'])
    rmse, mae, mape, rmse_mae = calculate_metrics(df_cv)
    return {'loss': rmse, 'status': STATUS_OK, 'other_metrics': (rmse, mae, mape, rmse_mae)}



# Función para ejecutar Hyperopt
def run_hyperopt(train_df, cutoff, max_evals=250):

    space = {
        'n_changepoints': hp.quniform('n_changepoints', 20, 100, 1),  # Genera flotantes que necesitan ser convertidos a enteros
        'changepoint_range': hp.uniform('changepoint_range', 0.7, 0.9),  # Corregido para usar la etiqueta correcta
        'seasonality_mode': hp.choice('seasonality_mode', ['additive', 'multiplicative']),
        'changepoint_prior_scale': hp.uniform('changepoint_prior_scale', 0.01, 0.9),
        'seasonality_prior_scale': hp.uniform('seasonality_prior_scale', 0.01, 100.0),
        'holidays_prior_scale': hp.uniform('holidays_prior_scale', 0.01, 100.0),
        'interval_width': hp.uniform('interval_width', 0.8, 0.95),
    }
    trials = Trials()
    best = fmin(fn=lambda params: objective_hyperopt(params, train_df, cutoff),
                space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    best_params = space_eval(space, best)  # Use space_eval to get actual parameter values
    best_metrics = trials.best_trial['result']['other_metrics']
    return best_metrics, best_params

# Función para comparar los optimizadores
def compare_optimizers(train_df, cutoff,trials=250):
    start_time = time.time()
    optuna_metrics, optuna_params = run_optuna(train_df, cutoff, n_trials=trials)
    optuna_time = time.time() - start_time

    start_time = time.time()
    hyperopt_metrics, hyperopt_params = run_hyperopt(train_df, cutoff, max_evals=trials)
    hyperopt_time = time.time() - start_time

    metrics = ['RMSE', 'MAE', 'MAPE', 'RMSE+MAE']
    results = pd.DataFrame({
        'Optimizer': ['Optuna', 'Hyperopt'],
        'Time (s)': [optuna_time, hyperopt_time]
    })
    for i, metric in enumerate(metrics):
        results[metric] = [optuna_metrics[i], hyperopt_metrics[i]]

    df_optuna = pd.DataFrame([optuna_params])
    df_hyperopt = pd.DataFrame([hyperopt_params])

    return results, df_optuna, df_hyperopt

def all_hyperparameters_tunning(train_df:pd.DataFrame,param_grid:Dict[Any, Any],cutoff:Dict[Any, Any]=None) -> pd.DataFrame:
    
    # Count Possible combinations of params
    grid = ParameterGrid(param_grid)
    cnt = 0
    for p in grid:
        cnt = cnt+1

    print('Total Possible Models',cnt)

    # Set cross validation parameters
    if cutoff:
        initial = cutoff['initial']
        period = cutoff['period']
        horizon = cutoff['horizon']
    else:
        initial =  str(round(msp.get_df_n_days(train_df)/1.5,0)) + " days"
        period = '10 days'
        horizon = '20 days'

    # Find the best parameters
    args = list(product(*param_grid.values()))

    df_ps_list = []

    for arg in tqdm(args):
        m = Prophet(*arg).fit(train_df)
        df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel="threads")
        df_p = performance_metrics(df_cv, rolling_window=1)
        df_p['params'] = str(arg)
        df_ps_list.append(df_p)

    df_ps = pd.concat(df_ps_list, ignore_index=True)
    df_ps['mae+rmse'] = df_ps['mae'] + df_ps['rmse']
    df_ps = df_ps.sort_values(['mae+rmse'])
    return df_ps