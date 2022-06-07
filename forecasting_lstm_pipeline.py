import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

def preprocess_df(df_path):
    dataframe = pd.read_csv(df_path, parse_dates = True, index_col = ['date'])
    return dataframe

def scaling(y, dataframe):
    scaler = MinMaxScaler(feature_range = (0, 1))
    y = scaler.fit_transform()
    return y

def scaler():
    scaler = MinMaxScaler(feature_range = (0, 1))
    return scaler

def series_split(n_lookback, n_forecast):
    X, y = [], []
    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback : i])
        y.append(y[i : i + n_forecast])
    X, y = np.array(X), np.array(y)
    return X, y

def build_lstm_model(n_lookback, n_forecast, dropout_pct):
    inputs = tf.keras.Input(shape = (n_lookback, 1))
    x = tf.keras.layers.LSTM(50, return_sequences = True)(inputs)
    x = tf.keras.layers.Dropout(dropout_pct)(x)
    x = tf.keras.layers.LSTM(50, return_sequences = True)(x)
    x = tf.keras.layers.Dropout(dropout_pct)(x)
    x = tf.keras.layers.LSTM(50, return_sequences = True)(x)
    x = tf.keras.layers.Dropout(dropout_pct)(x)
    outputs = tf.keras.layers.Dense(n_forecast)
    lstm_model = tf.keras.Model(inputs = inputs, outputs = outputs)
    return lstm_model

def build_lambda_model(n_lookback, n_forecast):
    inputs = tf.keras.Input(shape = (n_lookback, 1))
    x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis = 1))(inputs)
    x = tf.keras.layers.LSTM(128, activation = tf.nn.relu, return_sequences = True)(x)
    x = tf.keras.layers.LSTM(128, activation = tf.nn.relu, return_sequences = True)(x)
    x = tf.keras.layers.LSTM(32, activation = tf.nn.relu)(x)
    outputs = tf.keras.layers.Dense(n_forecast)(x)
    model_lambda = tf.keras.Model(inputs = inputs, outputs = outputs)
    return model_lambda

def define_callbacks(model_name, save_path = "model_experiments"):
    return tf.keras.callbacks.ModelCheckpoint(model_name, 
    save_best_only = True, monitor = 'val_loss')

def create_forecast_df(y, n_lookback, n_forecast, model, df):
    X_pred = y[- n_lookback:]
    y_pred = X_pred.reshape(1, n_lookback, 1)
    y_pred = model.predict(X_pred).reshape(-1, 1)
    y_pred = scaler().inverse_transform(y_pred)

    df_past = df[['Close']].reset_index()
    df_past.rename(columns = {'index': 'Date', 'Close': 'Actual'}, inplace = True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]

    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = y_pred.flatten()
    df_future['Actual'] = np.nan

    results = df_past.append(df_future).set_index('Date')
    return results

def plot_predictions(results):
    fig, ax = plt.subplots(figsize = (13, 5))
    ax.plot(results.index, results['Forecast'], marker = '.', linestyle = '-', label = "Forecasted Values")
    ax.plot(results.index, results['Actual'], marker = 'o', linestyle = '-', label = "Actual Values")
    ax.legend()
    return fig, ax

def get_pipeline(preprocess_df, scaling, series_split, build_lstm_model, 
build_lambda_model, define_callbacks, create_forecast_df, plot_predictions):
    pipeline = Pipeline([
        ('preprocess df', preprocess_df()),
        ('scaling', scaling()),
        ('series split', series_split()),
        ('build lstm model', build_lstm_model()),
        ('build lambda model', build_lambda_model()),
        ('define callbacks', define_callbacks()),
        ('create forecasted df', create_forecast_df()),
        ('plot predictions', plot_predictions())])
    return pipeline