from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import pandas as plt

def metrics(original, predictions):
    mae = mean_absolute_error(original, predictions)
    rmse = sqrt(mean_squared_error(original, predictions))
    
    print('MAE: %.2f' % mae)
    print('RMSE: %.2f' % rmse)

def forecastML(model, trainX, trainY, testX, testY):
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    return testY, predictions

def visualize(original, predictions, title):
    plt.plot(original, label = 'Value')
    plt.plot(predictions, label = 'Forecast')
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('PM2.5 concentration')
    plt.legend()
    plt.show()

def build_matrix(data, look_back = 5):
    x, y = [], []
    for i in range(len(data) - look_back - 1):
        x.append(data[i:i + look_back])
        y.append(data[look_back + i])
    return np.array(x), np.array(y)