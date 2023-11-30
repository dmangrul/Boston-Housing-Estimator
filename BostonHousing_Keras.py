#IMPORT THE LIBRARIES
from keras.datasets import mnist
from keras import models
from keras import layers
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from keras.datasets import boston_housing
 

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
# =============================================================================
# Standardize your input
# =============================================================================
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# =============================================================================
#     # Standardize the training dataset
#     #(and calculate the mean and standard deviation)
# =============================================================================
train_data = sc.fit_transform(train_data)
# =============================================================================
#     Use this mean and standard deviation
#     calculated in the training dataset to
#     standardize the test dataset
# =============================================================================
test_data = sc.transform(test_data)

model = models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape = (13,) ))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

history = model.fit(train_data, train_targets, epochs=100, batch_size=1)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

train_mse_score, train_mae_score = model.evaluate(train_data, train_targets)

mae_history = history.history['mae'] 

predicted_prices = model.predict(test_data)  

mae_calculated = np.sum(np.abs(predicted_prices - test_targets.reshape((102,1))))/len(predicted_prices)

fig, ax = plt.subplots()
ax.plot(range(1, len(mae_history) + 1), mae_history, 'bo', label='Mean Absolute Error')
# b is for "solid blue line"
ax.set(xlabel='Epochs', ylabel='MAE',
       title='Mean Absolute Error');
ax.legend()

print("Test MAE Value:", test_mae_score)
print("Verified Test MAE:", mae_calculated)
