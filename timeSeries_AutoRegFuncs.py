import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm

def read_csv(csv_path, index_col, parse_dates):
    df = pd.read_csv(csv_path, index_col = index_col, 
    parse_dates = parse_dates) #Set index column to the dates, and parse dates
    return df

def plt_DMYResample(df, column_name, figsize, marker, linestyle):
    y = df[column_name]
    fig, ax = plt.subplots(figsize = figsize)
    ax.plot(y, marker = marker, linestyle = linestyle)
    ax.plot(y.resample('D').mean(), marker = 'o', linestyle = linestyle)
    ax.plot(y.resample('M').mean(), marker = '.', linestyle = '-')
    ax.plot(y.resample('Y').mean(), marker = marker, linestyle = linestyle)
    ax.legend()

def find_firstDifference_shift(df, column_name):
    df['First Difference'] = df[column_name] - df[column_name].shift()
    return df['First Difference']

def find_logDifference(df, column_name):
    df['Log Difference'] = df[column_name].apply(lambda x: np.log(x))
    return df['Log Difference']

def ADF_test(timeSeries, dataDesc):
    print(" > Is the {} stationary? ".format(dataDesc))
    dftest = sm.tsa.adfuller(timeSeries.dropna(), autolag = 'AIC')
    print("Test Statistic: {}".format(dftest[0]))
    print("P-Value = {}".format(dftest[1]))
    print('Critical Values: ')
    for k, v in dftest[4].items():
        print('\t{}: {} - The data is {} stationary with {}% confidence'
        .format(k, v, 'not' if v<dftest[0] else '', 100-int(k[:-1])))

#Uses original time series df and the log difference to determine both
#the log variance and original time series variance simultaneously
def find_variance(df, timeSeries, window_size, logDifference):
    df['Original Variance'] = pd.Series(timeSeries).rolling(window = window_size).var()
    df['Log Variance'] = pd.Series(logDifference).rolling(window = window_size).var()
    return df['Original Variance'], df['Log Variance']

#Returns the values in order lag_corr then partial_corr
def lag_Par_Corr(logFirstDiff, fft):
    lag_corr = sm.tsa.stattools.acf(logFirstDiff.iloc[1:], fft = fft)
    partial_corr = sm.tsa.stattools.pacf(logFirstDiff.iloc[1:])
    return lag_corr, partial_corr

def decomposition(y_values, model, freq):
    decomposition = seasonal_decompose(y_values, model = model, freq = freq)
    fig = plt.figure(figsize = (20, 6))
    fig = decomposition.plot()

def plot_ARIMA(df, column_name, logDifference):
    model = sm.tsa.ARIMA(df[column_name], model = 'additive', freq = 30)
    results = model.fit(display=-1)
    df['Forcasted Values'] = results.fittedvalues
    fig, ax = plt.subplots(figsize = (14, 5))
    ax.plot(df['Forecasted Values'], marker = 'o', linestyle = '-.')
    ax.plot(logDifference, linestyle = '-', marker = '.')
    ax.legend()






