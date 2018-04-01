from unittest.mock import inplace

import quandl,math

#For loading dataset as a dataframe
import pandas as ps

#For Number crunching and calculations
import numpy as np

#For ML algorithms
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import  LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
from datetime import datetime
import pickle

style.use('ggplot')


def getDataFrame():
    #quandl.ApiConfig.api_key = 'Lwtdg2tcNqkxNA8JMnWs'
    # getting the data frame table frm quandl and manipulating it
    #df = quandl.get_table('WIKI/PRICES', ticker='A')

    #loading data frame from csv
    df = ps.read_csv('WIKI-PRICES.csv')

    df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
    df['hl_prC'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100
    df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100.0
    df = df[['adj_close', 'hl_prC', 'PCT_change', 'adj_volume']]
    return df;


def predictFutureLabel():
    df = getDataFrame()
    forecast_col = 'adj_close'

    df.fillna(-99999,inplace=True)

    forecast_out = int(math.ceil(.01 * len(df)))
    print("Forecast Number is: ",forecast_out)

    #The below statement shifts the value starting from the 100th row up( eg: 0th element will have the value of the 100th row and so on etc...)
    df['label'] = df[forecast_col].shift(-forecast_out)

    #Features all columns except Label
    X = np.array(df.drop(['label'],1))
    #Set the feature set leaving the last (forecast_out=)100 rows from the dataset
    X = X[:-forecast_out]

    #Get the feature set for the last (forecast_out=)100 rows from the data set for which we want to predict
    # because we have shifted the data for the last 100 rows as label for the rows above
    X_ToBeForecasted = X[-forecast_out:]

    #Feature scaling [x = (x - average of all x)/maxX - minX or Standard deviation]}
    X = preprocessing.scale(X)

    df.dropna(inplace=True)

    #The Label column
    Y = np.array(df['label'])

    #Prepare cross validation dataset
    x_train,x_test,y_train,y_test = cross_validation.train_test_split(X,Y,test_size=.02)

    #define the Training Algortithm
    clf = LinearRegression(n_jobs=-1)
    #LinearRegression()

    #feed the training data to the Training algorithm
    clf.fit(x_train,y_train)

    #Serializing trained classifier into a file
    with open('linearregression.pickle', 'wb') as f:
        pickle.dump(clf, f)

    # deSerializing trained classifier back into clf
    pickle_in = open('linearregression.pickle', 'rb')
    clf = pickle.load(pickle_in)

    #validate the training algortihm prediction accuracy with the test data [squared error= [Yprediction - Yactual]^2.sum() for all feature and label]
    accuracy = clf.score(x_test,y_test)

    print("Accuracy is:",accuracy)

    #Predictio of label for the last 100 rows of the data set using Linear model
    forecast_labels = clf.predict(X_ToBeForecasted)


    #print(forecast_labels,accuracy,forecast_out)

    return df,forecast_labels


df,forecast_labels = predictFutureLabel()
print(forecast_labels)
df['Forecast'] = np.nan


#Gets the name for the last Row of the data set
last_rec = df.iloc[-1].name
next_rec = last_rec + 1

#iterate over the forecast_labels to create dataset along with the older dataset,
#but with the Forecast column having values of the forecast_labels and not NAN
for i in forecast_labels:
    next_rec = next_rec
    next_rec += 1
    #create rows for the forecast_labels length and fill all columns with NAN but the forecast column with the forecast data
    df.loc[next_rec] = [np.nan for _ in range(len(df.columns)-1)]+[i]

print(df)

np.savetxt(r'output.txt', df.values)

df['adj_close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('name')
plt.ylabel('Price')
plt.show()