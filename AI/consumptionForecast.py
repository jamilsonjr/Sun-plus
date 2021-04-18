# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 12:46:55 2021

@author: asus
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from keras.layers import LSTM
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import datetime as dt
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# from matplotlib.pyplot import imshow, pause
# pause(1)

import warnings
warnings.filterwarnings("ignore")

import warnings
warnings.filterwarnings("ignore")
def calculateErrors (test_set, prediction_set):
    r2 = r2_score(test_set, prediction_set)
    rmse = sqrt(mean_squared_error(test_set, prediction_set))
    # print('r2 %.3f' % r2)
    print('RMSE %.3f' % rmse)
    
    return rmse

def AutoRegressiveForecastMethod (data_ptd, test_set, lags):
    # plot properties
    plt.rcParams['figure.figsize']=(20,10)
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size': 20})
    
    global coeficients

    X_ptd = data_ptd.dropna()
    train_data_ptd = X_ptd[0:len(X_ptd)]

    model_ptd = AutoReg(train_data_ptd, lags=lags)
    model_fitted_ptd = model_ptd.fit()
    predictions_ptd = model_fitted_ptd.predict(start=len(train_data_ptd), 
        end=len(train_data_ptd) + len(test_set)-1, 
        dynamic=False)
    predictions_ptd.index = test_set.index
    
    return predictions_ptd

def AutoRegressive_m2 (data, begin_date, end_day, days_to_forecast):
    
    # if  initialize.restrict_data == -2:
    #     lags = 96*7
    # else:
    #     lags = 96*15
    
    lags = 96*5
    
    data_ptd_restricted = data.loc[begin_date: end_day]
    # data_ptd_restricted = data_ptd.loc[begin_date: end_day]
    test_set = data.loc[end_day+pd.DateOffset(minutes=15):end_day+pd.DateOffset(days=days_to_forecast)]
    
    forecast_m2 = AutoRegressiveForecastMethod(data_ptd_restricted, test_set, lags)
    
    # concatenate true (test_set) with forecasted (prediction) to plot
    compare_df_ptd = pd.concat([test_set, forecast_m2], axis=1).rename(columns={'value': 'actual', 0:'predicted'})
    # compare_df_ptd.plot(title='Model 2 - Forecast AR')

    # calculate errors per day
    if days_to_forecast!=1:
        for i in range(0, 96*days_to_forecast, 96):
            print("Error in day", (1+i//96))
            calculateErrors (test_set[i:(i+96)], forecast_m2[i:(i+96)])
            print()
    
    rmse2 = calculateErrors (test_set, forecast_m2)
    
    return compare_df_ptd , rmse2 

dataset_tagus = pd.read_csv('2018-Taguspark.csv', skipfooter=4, engine = 'python')
dataset_ist = pd.read_csv('2018-Alameda.csv', skipfooter=4, engine='python')

dataset_ist = dataset_ist.rename(columns= {'Data hora': 'date', 'Activa kW': 'P', 'Reactiva Indutiva kVAR': 'Qi', 'Reactiva Capacitiva kVAR': 'Qc' })
# %% IST AR:
# dataset_ist['P'] = 0.1 * 
dataset_ist['date'] = pd.to_datetime(dataset_ist['date'])

# HERE
dataset_ist = dataset_ist.loc[dataset_ist['date'].dt.dayofweek >=5]

dataset_ist.set_index('date', inplace=True)
dataset_ist=dataset_ist.sort_values(by='date')

begin_date = '2018-01-01 00:15:00'
end_date = '2018-02-16 23:45:00'
end_day = dt.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
days_to_forecast = 1


# AR:
forecast_AR, rmse_AR = AutoRegressive_m2 (dataset_ist['P'], begin_date, end_day, days_to_forecast)
forecast_AR = forecast_AR.rename(columns={'P': 'real', 'predicted': 'predicted'})

plt.figure()
forecast_AR = forecast_AR.rename(columns = {'real': 'Real Consumption', 'predicted': 'Predicted Consumption'})
forecast_AR.plot(title='Forecast - Consumption with LSTM Model',lw=4).set(ylabel='kW', xlabel = 'Day')




# %% LSTM:
plt.rcParams['figure.figsize']=(20,10)
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20, 'lines.linewidth' : 2})

begin_date = '2018-01-01 00:15:00'
end_date = '2018-02-16 23:45:00'
end_day = dt.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
days_to_forecast = 1


end_date_dt = dt.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

number_day_train = 30
begin_date_dt = end_date_dt - pd.DateOffset(days=number_day_train) + pd.DateOffset(minutes=15)
begin_date = begin_date_dt.strftime('%Y-%m-%d %H:%M:%S')
df_original = pd.read_csv('2018-Alameda.csv', dayfirst=True, parse_dates=[0],skipfooter=4, usecols = ['Data hora', 'Activa kW'], engine='python')
df_original = df_original.rename(columns= {'Data hora': 'date', 'Activa kW': 'P'})

df_original=df_original.sort_values(by='date')
df = df_original[(df_original['date'] >=begin_date) & (df_original['date'] <=end_date)]

#HERE
df = df.loc[df['date'].dt.weekday >=5]

train_dates = pd.to_datetime(df['date'])

cols = list(df)[1]
df_for_training = df[cols].astype(float)
df_for_training = df_for_training.to_numpy()
df_for_training = df_for_training.reshape(-1,1)
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)

df_for_training_scaled = scaler.transform(df_for_training)

# training series
trainX = []

# predictions
trainY = []

n_future = 1
n_past = 96

for i in range (n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i-n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i+n_future - 1: i+n_future,0])


# convert trainX and trainY into array (before they were lists)
trainX, trainY = np.array(trainX), np.array(trainY) 

model = Sequential()
# 64 unidades
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences = False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(loss='mse', optimizer='adam')
print(model.summary())

#fit model
# epochs = number of times forward and backward
epochs = 20
batch_size = 32
history = model.fit(trainX, trainY, epochs, batch_size, validation_split = 0.1, verbose = 1)


n_future=97* days_to_forecast
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq = '15min').tolist()
forecast = model.predict(trainX[-n_future:], batch_size = 32) #forecast

# inverter escala de normalização
forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0] 

begin_time_test = end_date

end_time_test = end_date_dt + pd.DateOffset(days=1)
end_time_test = end_time_test.strftime('%Y-%m-%d %H:%M:%S')
# obter test set:
testar = df_original[(df_original['date']>=begin_time_test)&(df_original['date']<=end_time_test)]

pred_df = pd.DataFrame(y_pred_future, columns = ['predictions'])

pred_df.reset_index(drop=True, inplace = True)
testar.reset_index(drop=True, inplace = True)
forecast_LSTM = pd.concat([testar,pred_df], axis=1)
forecast_LSTM.index = forecast_period_dates
forecast_LSTM = forecast_LSTM.rename(columns = {'P': 'real', 'predictions': 'predicted'})

plt.figure()
forecast_LSTM.set_index('date', inplace=True)
forecast_LSTM.plot(title='Forecast - Consumption with LSTM Model',lw=4).set(ylabel='kW', xlabel = 'Day')

# %% RADIATION:
    

dataMeteo = pd.read_csv('sunlab-faro-meteo-2017.csv', sep=';', na_values=-1.500000e+09, usecols = ['Global Radiation [W/m2]', 'Datetime'])
dataMeteo = dataMeteo.sort_values(by=['Datetime'])
dataMeteo['Datetime'] = pd.to_datetime(dataMeteo['Datetime'], utc = True)
dataMeteo['Datetime'] = pd.to_datetime(dataMeteo['Datetime']).dt.tz_localize(None)
dataMeteo = dataMeteo.drop_duplicates(subset=['Datetime'])
dataMeteo['Datetime'] = dataMeteo['Datetime'] + pd.DateOffset(years=1)
dataMeteo = dataMeteo.rename(columns = {'Datetime': 'date', 'Global Radiation [W/m2]':'radiation'})

dataMeteo.set_index('date', inplace=True)

begin_date_radiation_dt = dt.datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S') + pd.DateOffset(minutes=15)
end_date_radiation_dt = begin_date_radiation_dt + pd.DateOffset(days=1) - pd.DateOffset(minutes=15)
begin_date_radiation = begin_date_radiation_dt.strftime('%Y-%m-%d %H:%M:%S')
end_date_radiation = end_date_radiation_dt.strftime('%Y-%m-%d %H:%M:%S')

dataMeteo = dataMeteo.loc[begin_date_radiation: end_date_radiation]
# Net = Natural - PV <=> Net = Natural -k*radiation

# Natural = sem PV
# Net = com PV

MeteoMinute00 = dataMeteo[dataMeteo.index.minute == 00]
MeteoMinute15 = dataMeteo[dataMeteo.index.minute == 15]
MeteoMinute30 = dataMeteo[dataMeteo.index.minute == 30]
MeteoMinute45 = dataMeteo[dataMeteo.index.minute == 45]

trialMeteo = pd.concat([MeteoMinute00, MeteoMinute15, MeteoMinute30, MeteoMinute45])
trialMeteo = trialMeteo.sort_values(by=['date'])

trialMeteo=trialMeteo * 2

final_plots = pd.concat([forecast_AR, trialMeteo['radiation']],axis=1)
final_plots = final_plots.rename(columns= {'radiation': 'PV'})
final_plots['PV'].fillna(0, inplace=True)
final_plots = final_plots.sort_values(by=['date'])
# final_plots = final_plots.drop(columns ='real')
final_plots['Surplus'] = final_plots['PV'] - final_plots['predicted']
final_plots = final_plots.rename(columns = {'predicted': 'Predicted Consumption'})
final_plots.plot(title = 'Surplus of Energy',lw=4, ylim = [0,1600]).set (ylabel='kW')

