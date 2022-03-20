# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 16:55:32 2020

New York City Taxi Fare Prediction

In this playground competition, hosted in partnership with Google Cloud and Coursera, 
you are tasked with predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given 
the pickup and dropoff locations. While you can get a basic estimate based on just the distance between the two points, 
this will result in an RMSE of $5-$8, depending on the model used (see the starter code for an example of this approach 
in Kernels). Your challenge is to do better than this using Machine Learning techniques!

To learn how to handle large datasets with ease and solve this problem using TensorFlow, 
consider taking the Machine Learning with TensorFlow on Google Cloud Platform specialization on Coursera 
-- the taxi fare problem is one of several real-world problems that are used as case studies in the series of courses. 
To make this easier, head to Coursera.org/NEXTextended to claim this specialization for free for the first month!

@author: Admin
"""

%matplotlib qt

#Importing libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from scipy import stats
import math

#Importing data
df_train = pd.read_csv('Data/train.csv', nrows = 100000)

describe = df_train.describe()
futures = df_train.columns

#Making new futures
def making_new_futures(df_train):
    df_train['distance'] = ((df_train['dropoff_longitude']-df_train['pickup_longitude'])**2 + (df_train['dropoff_latitude']-df_train['pickup_latitude'])**2)**(1/2)
    df_train['abs_longitude'] = abs(df_train['dropoff_longitude']-df_train['pickup_longitude'])
    #df_train['abs_latitude'] = abs(df_train['dropoff_latitude']-df_train['pickup_latitude'])
    
    '''Dealing with time'''
    df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
    df_train['hour'] = df_train['pickup_datetime'].dt.hour
    df_train['day_of_the_week'] = df_train['pickup_datetime'].dt.dayofweek
    df_train['month'] = df_train['pickup_datetime'].dt.month
    df_train['year'] = df_train['pickup_datetime'].dt.year
    
    #Split for day and night
    df_train.loc[(df_train['hour'] >= 6) & (df_train['hour'] <22), 'is_day'] = 1
    df_train['is_day'] = df_train['is_day'].fillna(0)
    #Split for weekend or not 
    df_train.loc[(df_train['day_of_the_week'] >= 5), 'is_weekend'] = 1
    df_train.loc[(df_train['day_of_the_week'] < 5), 'is_weekend'] = 0
    return df_train

#Improve formule for distance beetween points
def calculate_distance(lon1, lon2, lat1, lat2):
    lat = (lat1 + lat2) / 2 * 0.01745
    dx = 111.3 * math.cos(lat) * (lon1 - lon2)
    dy = 111.3 * (lat1 - lat2)
    distance = math.sqrt(dx * dx + dy * dy)
    return distance

'''distance_tab = []
for data in df_train.iloc:
    distance_tab.append(calculate_distance(data['pickup_longitude'], data['dropoff_longitude'], data['pickup_latitude'], data['dropoff_latitude']))'''
    


#Cleaing data
def cleaing_data(df_train):
    #Drop row where value equal '0'
    df_train = df_train[(df_train != 0).all(1)]
    #Drop value where distance > 5
    df_train = df_train[df_train.distance<0.5]
    df_train = df_train[df_train.distance>0.001]
    #Drop value where fare < 0
    df_train = df_train[df_train.fare_amount>1]
    #Drop value where passenger = 0
    df_train = df_train[df_train.passenger_count>0]
    
    drop_list = ['key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'hour', 'day_of_the_week', 'passenger_count']
    df_train = df_train.drop(drop_list, axis=1)
    
    return df_train


#histogram and normal probability plot
def histogram_plot(var=''):
    sns.distplot(df_train[var], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train[var], plot=plt)
    
#plot_histogram('distance')


#box plot passenger_count/fare_amount
def box_plot(var=''):
    data = pd.concat([df_train['fare_amount'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="fare_amount", data=data)
    
#box_plot('passenger_count')

#scatter plot X/fare_amount
def scatter_plot(var=''):
    data = pd.concat([df_train['fare_amount'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='fare_amount')

#scatterplot
sns.set()
cols = ['fare_amount', 'distance', 'abs_longitude', 'abs_latitude', 'passenger_count']
sns.pairplot(df_train[cols], size = 2.5)
plt.show()


#plot start and end point
def plot_points(row=0):
    point = df_train.loc[row]
    plt.scatter(x=point['pickup_longitude'], y=point['pickup_latitude'], label='Start')
    plt.scatter(x=point['dropoff_longitude'], y=point['dropoff_latitude'], label='End')
    plt.title(point['fare_amount'])
    plt.legend()
    plt.show()
    
plot_points(1)

#Checking on correlation 
corr = df_train.corr(method='pearson')

#applying log transformation
def log_transform(df_train):
    #Fare_amount
    #df_train['fare_amount'] = np.log(df_train['fare_amount'])
    #histogram_plot('fare_amount')
    #Distance
    df_train['distance'] = np.log(df_train['distance'])
    #histogram_plot('distance')
    #Abs_longitude
    df_train['abs_longitude'] = np.log(df_train['abs_longitude'])
    #histogram_plot('abs_longitude')
    #Abs_longitude
    #df_train['abs_latitude'] = np.log(df_train['abs_latitude'])
    #histogram_plot('abs_latitude')
    return df_train
    

'''Prepare data for training'''
def prepare_data(data):
    #New futures
    data = making_new_futures(data)
    #Cleaning data
    data = cleaing_data(data)
    #Log transform
    data = log_transform(data)
    return data
    
raw_data = pd.read_csv('Data/train.csv', nrows = 100000)
df_train = prepare_data(raw_data)


'''Making model'''
import tensorflow as tf
from tensorflow import keras

n_inputs = 6
n_hidden1 = 300
n_hidden2 = 150
n_output = 1

#Making a struture of the model 
model = keras.Sequential([
    keras.layers.InputLayer(n_inputs), #input layer
    keras.layers.Dense(n_hidden1, activation='elu'), #hidden layer 1
    keras.layers.Dense(n_hidden2, activation='elu'), #hidden layer 2
    keras.layers.Dense(n_output, activation='linear'), #output layer
    ])

model.compile(loss='mae',
              optimizer='adam',
              metrics=['mae'])

#Fitting 
y_label = df_train.drop('fare_amount')

history = model.fit(df_train, y_label, epochs=100, validation_split=0.2)


