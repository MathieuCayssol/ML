#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 20:07:51 2019

@author: mathieucayssol
"""


# Matplotlib
import matplotlib.pyplot as plt
# Tensorflow
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Numpy and Pandas
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



dataframe = pd.read_csv('BTC-USD.csv')


dataframe['Date'] = pd.to_datetime(dataframe['Date'])
my_data = dataframe[['Date', 'Open']]
#print(my_data[0:1156])
my_data.set_index('Date', inplace=True)


#func_vwap = lambda x: pd.np.sum(x.price*x.homeNotional)/pd.np.sum(x.homeNotional)
#df_vwap = my_data.groupby(pd.Grouper(freq="1Min")).apply(func_vwap)


len_data = int(len(my_data))
len_val = int(0.6*len_data)
len_test = int(0.8*len_data)



df_train = my_data[0:len_test]
#df_val = my_data[len_val-60:len_test]
df_test = my_data[len_test-60:len_data]



sc = MinMaxScaler()
df_train_sc = sc.fit_transform(df_train['Open'].values.reshape(-1,1)) # Values for select just vwap 
#df_val_sc = sc.transform(df_val['Open'].values.reshape(-1,1)) # reshape(-1,1) for create a column vector
df_test_sc = sc.transform(df_test['Open'].values.reshape(-1,1))



def lstm_data(x,sequence_len):
    x_final, y_final = [], []
    for i in range(sequence_len, len(x)-1):
        x_i = x[i-sequence_len:i]
        y_i = x[i:i+2]
        x_final.append(x_i)
        y_final.append(y_i)
        
    x_final = np.array(x_final).reshape(-1,sequence_len,1)
    y_final = np.array(y_final)
    return x_final, y_final



time_step = 60

#tab = [2,3,4,5,6,7,8]

x_train, y_train = lstm_data(df_train_sc,time_step)



#print(a[0:10], b[0:10])

model = Sequential()
model.add(LSTM(units=50, activation= 'relu', return_sequences = True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation= 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation= 'relu', return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units=120, activation= 'relu'))
model.add(Dropout(0.3))

model.add(Dense(units=2))

# model optimizer 



model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=100)

#plt.plot(df_vwap.index, df_vwap.iloc[:,0]) # first column of data frame
#plt.show()


x_test, y_test = lstm_data(df_test_sc,time_step)

y_pred = model.predict(x_test)

# Back scale data

y_test = sc.inverse_transform(np.array(y_test).reshape(1,-1))
y_pred = sc.inverse_transform(np.array(y_pred).reshape(1,-1))


y_test = y_test.reshape(-1)
y_pred = y_pred.reshape(-1)



plt.plot(y_test[50:300])
plt.plot(y_pred[50:300])

plt.show()








