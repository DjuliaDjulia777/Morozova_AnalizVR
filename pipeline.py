import pandas as pd
import numpy as np
import lightgbm as lgb

def create_features(series, lags=[1,2,3,6,12,24,48,168]):
    df_feat = pd.DataFrame(series)
    col_name = df_feat.columns[0]
    for lag in lags:
        df_feat[f'lag_{lag}'] = df_feat[col_name].shift(lag)
    df_feat['rolling_mean_6'] = df_feat[col_name].shift(1).rolling(6).mean()
    df_feat['rolling_std_6']  = df_feat[col_name].shift(1).rolling(6).std()
    df_feat['rolling_mean_24'] = df_feat[col_name].shift(1).rolling(24).mean()
    df_feat['rolling_std_24'] = df_feat[col_name].shift(1).rolling(24).std()
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    df_feat['month'] = df_feat.index.month
    return df_feat

def forecast_pipeline(train_series, horizon=168):
    train_series = train_series.copy()
    target_col = train_series.name
    feat = create_features(train_series).dropna()
    X_train = feat.drop(target_col, axis=1)
    y_train = feat[target_col]
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    history = train_series.copy()
    predictions = []
    last_date = history.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=horizon, freq='h')
    
    for step, current_date in enumerate(future_dates):
        history = pd.concat([history, pd.Series(np.nan, index=[current_date], name=target_col)])
        feats = create_features(history)
        X_step = feats.drop(target_col, axis=1).iloc[[-1]]
        pred = model.predict(X_step)[0]
        history[current_date] = pred
        predictions.append(pred)
    return pd.Series(predictions, index=future_dates, name='Prediction'), model

from google.colab import files
files.download('pipeline.py')
