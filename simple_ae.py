""" Code to create simple AE for TS"""
"""Main"""
import pandas as pd

"""Data Viz"""
import seaborn as sns

color = sns.color_palette()

"""Data preparation and model evaluation"""
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split

"""Algos"""

"""TensorFlow and Keras"""
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras import regularizers

""" import functions"""
from visualizations import plotResults
from model_functions import anomalyScores

data = pd.read_csv("/home/anjan/work/datasets/creditcard.csv")
dataX = data.copy().drop(['Class', 'Time'], axis=1)
dataY = data['Class'].copy()
featuresToScale = dataX.columns
sX = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])

X_train, X_test, y_train, y_test = \
    train_test_split(dataX, dataY, test_size=0.33, \
                     stratify=dataY)
X_train_AE = X_train.copy()
X_test_AE = X_train_AE.copy()

num_epochs = 10
batch_size = 32

# # building the models
# model = Sequential()
# model.add(Dense(units=29,activation='linear',input_dim=29))
# model.add(Dense(units=29, activation='linear'))

# building non-linear under complete auto encoder
model = Sequential()
model.add(Dense(units=40, activation='linear', \
                activity_regularizer=regularizers.l1(10e-5), input_dim=29))
model.add(Dropout(0.05))
model.add(Dense(units=29, activation='linear'))

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

# fitting the model
history = model.fit(x=X_train_AE, y=X_train_AE,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_train_AE, X_train_AE),
                    verbose=1)

# evaluating the model
predictions = model.predict(X_test, verbose=1)
anomalyScoresAE = anomalyScores(X_test, predictions)
preds = plotResults(y_test, anomalyScoresAE, True)
