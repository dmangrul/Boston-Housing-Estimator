#IMPORT THE LIBRARIES
from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def readDataFile():
    trainingFile = 'boston.csv'
    print("training file: ", trainingFile)
    raw_train = open(trainingFile, 'rt')

    data_train = np.loadtxt(raw_train, skiprows = 1, delimiter=",")

    return data_train


data = readDataFile()

x = data[:, :12]
y = data[:,-1]

import sklearn.model_selection as model_selection
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.75,test_size=0.25, random_state=101)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_temp = x_train
# Standardize the training dataset 
#(and calculate the mean and standard deviation)
x_train = sc.fit_transform(x_train)

#Use this mean and standard deviation 
#calculated in the training dataset to 
#standardize the test dataset
x_test = sc.transform (x_test)

from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(x_train, y_train)

predicted_prices = model.predict(x_test)  

mae = np.sum(np.abs(predicted_prices - y_test.reshape((1, len(y_test)))))/len(predicted_prices)

plt.scatter(y_test, predicted_prices/y_test)

print("Test MAE:", mae)
