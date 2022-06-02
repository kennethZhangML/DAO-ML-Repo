import numpy as np
import itertools 
import statsmodels as sm


def find_best_params(best_aic, best_pdq, best_seasonal_pdq, temp_model):
    best_seasonal_pdq = None
    best_aic = np.inf
    best_pdq = None
    temp_model = None

    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    for param in pdq:
        for seasonal_params in seasonal_pdq:
            temp_model = sm.tsa.SARIMAX(train, order = param, seasonal_order = seasonal_params, enforce_invertibility = False, enforce_stationarity = False)
            temp_model_results = temp_model.fit(disp = False)
        
        if temp_model_results.aic < best_aic:
            best_aic = temp_model_results.aic
            best_pdq = param
            best_seasonal_pdq = seasonal_pdq
    return best_seasonal_pdq, best_aic, best_pdq