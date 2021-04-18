# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:11:40 2021

@author: asus
"""

import pandas as pd
from pandas import read_csv
from datetime import datetime
import numpy as np
from vmdpy import VMD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, UpSampling1D
from tensorflow.python.keras.layers.core import Reshape
from math import sqrt
# Ao meio-dia quero prever no dia a seguir (shift = 12)
# input time de 4 = 1 hora de data -> dados de input do modelo

# Em time steps
INPUT_TIME = 64
# 6 horas
HOURS_PREDICT = 24
SHIFT = 0

LABEL_COLUMN = 'Ghi'

# data = subsinais criados com VMD
 
def SplitData(data, original):
  i = 0
  x_temp = []
  y_temp = []
  baseline_temp = []
  while i + INPUT_TIME + HOURS_PREDICT + SHIFT < data.shape[1]:
      x_temp.append(data[:,i:i+INPUT_TIME].reshape(data[:,i:i+INPUT_TIME].shape[1], data[:,i:i+INPUT_TIME].shape[0]))
      y_temp.append(original[i+INPUT_TIME+SHIFT:i+INPUT_TIME+SHIFT+HOURS_PREDICT])
      baseline_temp.append(original[i+INPUT_TIME-24:i+INPUT_TIME])
      i += 1
  return np.array(x_temp, dtype=np.float32), np.array(y_temp, dtype=np.float32), np.array(baseline_temp, dtype=np.float32)

def avaliateVMD(df):
    mape = []
    for i in range(2, 25):
        u_temp, u_hat, omega = VMD(df[LABEL_COLUMN][0:10000], 2000, 0., i, 0, 1, 1e-7)
        mape_temp = 0
        for t in range(u_temp.shape[1]):
            if df[LABEL_COLUMN][t] < 1:
                continue
            sum1 = 0
            for k in range(u_temp.shape[0]):
                sum1 += u_temp[k][t]
            mape_temp += abs((df[LABEL_COLUMN][t] - sum1)/df[LABEL_COLUMN][t])
        mape.append(mape_temp/len(u_temp[0]))
        print(str(i)+ " wavelets, MAPE: " + str(round(mape[i-2]*100, 2)) +"%")

def cnn(train_x, train_y):
	# prepare data
	# define parameters
    epochs, batch_size, patience = 10, 64, 1500
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
	# define model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=2, activation='relu', strides = 2, input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters = 64, kernel_size=2, activation='relu', strides = 2))
    model.add(Conv1D(filters = 128, kernel_size=2, activation='relu', strides = 2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Reshape([8,16]))
    model.add(UpSampling1D(2))
    model.add(Conv1D(filters = 8, kernel_size=1, activation='relu', strides = 1))
    model.add(UpSampling1D(2))
    model.add(Conv1D(filters = 1, kernel_size=9, activation='relu', strides = 1))
    model.add(Reshape([24,1]))
    # model.add(Dense(n_outputs))
    
    model.compile(loss='mse', optimizer='adam')

    model.summary()
	# fit network
    # early_stopping = tf.keras.callbacks.EarlyStopping(patience=patience, mode='min', restore_best_weights= True)
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)
    return model


def calculate_errors(predictions, labels, baselines, max_, min_):
    mape = 0
    rmse = 0
    mae = 0
    mape_baseline = 0
    rmse_baseline = 0
    mae_baseline = 0
    removeCount = 0

    labels = labels*(max_-min_) + min_
    predictions = predictions*(max_-min_) + min_
    baselines = baselines*(max_-min_) + min_
    for prediction, label, baseline in zip(predictions, labels, baselines):
        for p, l, b in zip(prediction, label, baseline):
            rmse += pow(p - l, 2)
            rmse_baseline += pow(b - l, 2)
            mae += abs(l-p)
            mae_baseline += abs(l-b)
            if l != 0:
                if isinstance(p, np.ndarray):
                    p = p[0]
                mape += abs((l-p)/l)
                mape_baseline += abs((l-b)/l)
            else:
                removeCount += 1
    rmse = sqrt(rmse/(predictions.shape[0]*predictions.shape[1]))
    mape = (mape*100)/(predictions.shape[0]*predictions.shape[1] - removeCount)
    mae = mae/(predictions.shape[0]*predictions.shape[1])
    rmse_baseline = sqrt(rmse_baseline/(predictions.shape[0]*predictions.shape[1]))
    mape_baseline = (mape_baseline*100)/(predictions.shape[0]*predictions.shape[1] - removeCount)
    mae_baseline = mae_baseline/(predictions.shape[0]*predictions.shape[1])
    print("RMSE: " + str(round(rmse, 2)) + " MAE: " + str(round(mae, 2)) + " MAPE: " + str(round(mape)) + "%" + " Baseline RMSE: " + str(round(rmse_baseline, 2)) + " Baseline MAE: " + str(round(mae_baseline, 2)) + " Baseline MAPE: " + str(round(mape_baseline)) + "%")

dataset = pd.read_csv('Solcast_test_file.csv', usecols=['PeriodEnd','Ghi'])

dataset['PeriodEnd'] = pd.to_datetime(dataset['PeriodEnd'])
dataset.set_index('PeriodEnd')
dataset=dataset.sort_values(by='PeriodEnd')

# avaliateVMD(dataset)
order = 9
n = len(dataset)
len_train = int(n*0.8) 
train_original = dataset[LABEL_COLUMN][int(len_train*0.92):len_train]
test_original = dataset[LABEL_COLUMN][len_train:int(1.05*len_train)]

u_temp_train, u_hat_train, omega_train = VMD(train_original, 2000, 0., order, 0, 1, 1e-7)
u_temp_test, u_hat_test, omega_test = VMD(dataset[LABEL_COLUMN][len_train:int(1.05*len_train)], 2000, 0., order, 0, 1, 1e-7)

split_train_x, split_train_y, split_train_baseline = SplitData(u_temp_train, dataset[LABEL_COLUMN][0:len_train])
split_test_x, split_test_y, split_test_baseline = SplitData(u_temp_test, dataset[LABEL_COLUMN][len_train:])

train_df = u_temp_train
test_df = u_temp_test



# ve max e min de cada um dos novos sinais 
train_max = [element.max() for element in train_df]
train_min = [element.min() for element in train_df]
train_original_max = train_original.max()
train_original_min = train_original.min()

train = np.array([(train_ - min_) / (max_ - min_) for train_, min_, max_ in zip(train_df, train_max, train_min)])
test = np.array([(test_ - min_) / (max_ - min_) for test_, min_, max_ in zip(test_df, train_max, train_min)])
train_original = (train_original - train_original_min)/(train_original_max-train_original_min)
test_original = (test_original - train_original_min)/(train_original_max-train_original_min)

train_x, train_y, train_baseline = SplitData(train, train_original)
train_y = train_y.reshape(train_y.shape[0], train_y.shape[1])

train_baseline = train_baseline.reshape(train_baseline.shape[0], train_baseline.shape[1])

test_x, test_y, test_baseline = SplitData(test, test_original)
test_y = test_y.reshape(test_y.shape[0], test_y.shape[1])
test_baseline = test_baseline.reshape(test_baseline.shape[0], test_baseline.shape[1])

model = cnn(train_x, train_y)

predictions_train = model.predict(train_x)
predictions_train = predictions_train.reshape(predictions_train.shape[0], predictions_train.shape[1])
predictions_test = model.predict(test_x)
predictions_test = predictions_test.reshape(predictions_test.shape[0], predictions_test.shape[1])

# labels = realidade 
# x = o que entra
# y = o que devia sair  = labels
# # predictions = o que sai
# baseline -> considerar que radiancia de hoje é igaul à de ontem
# train_original_max e train_original_min para a desnormalizaçao dos dados de previsoes e labels

calculate_errors(predictions_train, train_y, train_baseline, train_original_max, train_original_min)
calculate_errors(predictions_test, test_y, test_baseline, train_original_max, train_original_min)
