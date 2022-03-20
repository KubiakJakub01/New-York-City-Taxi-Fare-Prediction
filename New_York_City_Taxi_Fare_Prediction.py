'''
New York City Taxi Fare Prediction
'''


#%matplotlib qt

'''Importing lib'''
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

'''
Preprocessing
'''

def clean(df):
    #Drop row where value equal '0'
    df = df[(df != 0).all(1)]
    # Delimiter lats and lons to NY only
    df = df[(-76 <= df['pickup_longitude']) & (df['pickup_longitude'] <= -72)]
    df = df[(-76 <= df['dropoff_longitude']) & (df['dropoff_longitude'] <= -72)]
    df = df[(38 <= df['pickup_latitude']) & (df['pickup_latitude'] <= 42)]
    df = df[(38 <= df['dropoff_latitude']) & (df['dropoff_latitude'] <= 42)]
    # Remove possible outliers
    df = df[(1 < df['fare_amount']) & (df['fare_amount'] <= 250)]
    # Remove inconsistent values
    df = df[(df['dropoff_longitude'] != df['pickup_longitude'])]
    df = df[(df['dropoff_latitude'] != df['pickup_latitude'])]
    
    return df

def add_new_futures(df):
    #Distance
    df = add_distances_features(df)
    #Time
    df = add_time_future(df)
    
    return df    
    
def add_distances_features(df):
    # Add distances from airpot and downtown
    ny = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)
    
    lat1 = df['pickup_latitude']
    lat2 = df['dropoff_latitude']
    lon1 = df['pickup_longitude']
    lon2 = df['dropoff_longitude']
    
    df['Manh_disctance'] = distance_manh(lat1,lat2,lon1,lon2)
    #df['Haversine_distance'] = distance_haversine(lat1,lat2,lon1,lon2)
    #df['distance'] = ((df['dropoff_longitude']-df['pickup_longitude'])**2 + (df['dropoff_latitude']-df['pickup_latitude'])**2)**(1/2)
    
    df['downtown_pickup_distance'] = distance_manh(ny[1], ny[0], lat1, lon1)
    df['downtown_dropoff_distance'] = distance_manh(ny[1], ny[0], lat2, lon2)
    df['jfk_pickup_distance'] = distance_manh(jfk[1], jfk[0], lat1, lon1)
    df['jfk_dropoff_distance'] = distance_manh(jfk[1], jfk[0], lat2, lon2)
    df['ewr_pickup_distance'] = distance_manh(ewr[1], ewr[0], lat1, lon1)
    df['ewr_dropoff_distance'] = distance_manh(ewr[1], ewr[0], lat2, lon2)
    df['lgr_pickup_distance'] = distance_manh(lgr[1], lgr[0], lat1, lon1)
    df['lgr_dropoff_distance'] = distance_manh(lgr[1], lgr[0], lat2, lon2)
    
    drop_list = ['pickup_latitude', 'dropoff_latitude', 'pickup_longitude', 'dropoff_longitude']
    df = df.drop(drop_list, axis=1)
    
    return df
 
# To Compute Haversine distance  
def distance_manh(pickup_latitude, pickup_longitude,dropoff_latitude,dropoff_longitude):
    abs_diff_longitude = (dropoff_longitude - pickup_longitude).abs()*80
    abs_diff_latitude = (dropoff_latitude - pickup_latitude).abs()*111

    meas_ang = 0.506
    
    Euclidean = (abs_diff_latitude**2 + abs_diff_longitude**2)**0.5 ### as the crow flies  
    delta_manh_long = (Euclidean*np.sin(np.arctan(abs_diff_longitude / abs_diff_latitude)-meas_ang)).abs()
    delta_manh_lat = (Euclidean*np.cos(np.arctan(abs_diff_longitude / abs_diff_latitude)-meas_ang)).abs()
    manh_length = delta_manh_long + delta_manh_lat
    
    return manh_length

# To Compute Haversine distance
def distance_haversine(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff coordinates.
    """
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))

def add_time_future(df):
    #Time
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_the_week'] = df['pickup_datetime'].dt.dayofweek
    df['month'] = df['pickup_datetime'].dt.month
    df['year'] = df['pickup_datetime'].dt.year
    df['night'] = df.apply (lambda x: is_night(x), axis=1)
    df['is_week'] = df.apply(lambda x: is_week(x), axis=1)
    # Drop 'pickup_datetime' as we won't need it anymore
    df = df.drop(['pickup_datetime','hour','day_of_the_week','month','year'], axis=1)
    
    return df

def is_night(row):
    if (row['hour'] <= 6) or (row['hour'] >= 22):
        return 1
    else:
        return 0

def is_week(row):
    if (row['day_of_the_week'] == 5 or row['day_of_the_week'] == 6):
        return 1
    else:
        return 0

        
def prepare_Date(df):
    df = clean(df)
    df = add_new_futures(df)
    return df

'''Making Model'''
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, regularizers
from tensorflow.keras.layers import Dense, BatchNormalization

def make_model(input_shape):
    #n_hidden0 = 512
    n_hidden1 = 256
    n_hidden2 = 128
    n_hidden3 = 64
    n_hidden4 = 32
    n_hidden5 = 8
    l1 = 0.001
    
    model = Sequential()
    #model.add(Dense(n_hidden0, activation='elu', input_dim=input_shape, activity_regularizer=regularizers.l1(l1)))
    #model.add(BatchNormalization())
    model.add(Dense(n_hidden1, activation='elu', input_dim=input_shape, activity_regularizer=regularizers.l1(l1)))
    #model.add(BatchNormalization())
    model.add(Dense(n_hidden2, activation='elu'))
    #model.add(BatchNormalization())
    model.add(Dense(n_hidden3, activation='elu'))
    #model.add(BatchNormalization())
    model.add(Dense(n_hidden4, activation='elu'))
    #model.add(BatchNormalization())
    model.add(Dense(n_hidden5, activation='elu'))
    #model.add(BatchNormalization())
    model.add(Dense(1))
    
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    
    return model

'''Importing data'''
# Load values in a more compact form
TRAIN_PATH = 'Data/train.csv'
TEST_PATH = 'Data/test.csv'
DATASET_SIZE = 1000000

datatypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

# Load all data
'''chunksize = 5000000
usecols = [2,3,4,5,6]
df_list = [] # list to hold the batch dataframe

for df_chunk in tqdm(pd.read_csv(TRAIN_PATH, usecols=usecols, dtype=datatypes, chunksize=chunksize)):
     
    # Neat trick from https://www.kaggle.com/btyuhas/bayesian-optimization-with-xgboost
    # Using parse_dates would be much slower!
    df_chunk['pickup_datetime'] = df_chunk['pickup_datetime'].str.slice(0, 16)
    df_chunk['pickup_datetime'] = pd.to_datetime(df_chunk['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    # Can process each chunk of dataframe here
    # clean_data(), feature_engineer(),fit()
    df_chunk = prepare_Date(df_chunk)
    df_chunk = scale_Date(df_chunk)
    
    # Alternatively, append the chunk to list and merge all
    df_list.append(df_chunk) 
    
# Merge all dataframes into one dataframe
all_data = pd.concat(df_list)

# Delete the dataframe list to release memory
del df_list'''
    
df_train = pd.read_csv(TRAIN_PATH, nrows=DATASET_SIZE, dtype=datatypes, usecols=[1,2,3,4,5,6])
all_data = prepare_Date(df_train)

del df_train

#corr = df_train.corr(method='pearson')

'''Prepering data to fit'''
#Split data
train_df, test_df = train_test_split(all_data, test_size=0.001, random_state=1)

# Get labels
train_labels = train_df['fare_amount'].values
test_labels = test_df['fare_amount'].values
train_df = train_df.drop(['fare_amount'], axis=1)
test_df = test_df.drop(['fare_amount'], axis=1)

# Scale data
def scale_Date(df):
    scaler = preprocessing.MinMaxScaler()
    df = scaler.fit_transform(df)
    return df

train_df = scale_Date(train_df)
test_df = scale_Date(test_df)



'''Training model'''
BATCH_SIZE = 256
EPOCHS = 5
n_input = train_df.shape[1]

model = make_model(n_input)

history = model.fit(x=train_df, y=train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.01,
                    verbose=1, 
                    shuffle=True)

test_loss, test_acc = model.evaluate(test_df, test_labels, verbose=1)
y_pred = model.predict(test_df)

print('Dokladnosc: ', test_acc)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor

regressor_RFR = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor_RFR.fit(train_df, train_labels)
y_pred = regressor_RFR.predict(test_df)

#Check the evaluation
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, y_pred))


'''TPOT'''
from sklearn.model_selection import RepeatedKFold
from tpot import TPOTRegressor
# define evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
model = TPOTRegressor(generations=5, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
# perform the search
model.fit(train_df, train_labels)
# score model
model.score(test_df, test_labels)
y_pred = model.predict(test_df)
# export the best model
model.export('tpot_best_model.py')

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive

'''Model created by TPOT'''
def tpot_model(X, y):
    exported_pipeline = make_pipeline(
    VarianceThreshold(threshold=0.0005),
    PCA(iterated_power=8, svd_solver="randomized"),
    RandomForestRegressor(bootstrap=True, max_features=0.4, min_samples_leaf=2, min_samples_split=4, n_estimators=100)
    )
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 1)
    
    exported_pipeline.fit(X, y)
    
    return exported_pipeline
   
model = tpot_model(train_df, train_labels)
y_pred = model.predict(test_df)

#Check the evaluation
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, y_pred))

