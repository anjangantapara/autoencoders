""" Code to create simple AE for TS

"""
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout, Dense

from model_functions import anomaly_scores
from visualizations import plot_results

# color = sns.color_palette()

data = pd.read_csv("/home/anjan/work/datasets/creditcard.csv")
dataX = data.copy().drop(['Class', 'Time'], axis=1)
dataY = data['Class'].copy()
featuresToScale = dataX.columns
sX = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
dataX.loc[:, featuresToScale] = sX.fit_transform(dataX[featuresToScale])

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, stratify=dataY)
X_train_AE = X_train.copy()
X_test_AE = X_train_AE.copy()

num_epochs = 10
batch_size = 32

# # building the models
# model = Sequential()
# model.add(Dense(units=29,activation='linear',input_dim=29))
# model.add(Dense(units=29, activation='linear'))

# building non-linear under complete auto encoder
stacked_encoder = Sequential([
    Dense(units=29, activation='linear', activity_regularizer=regularizers.l1(10e-5), input_dim=29),
    Dropout(0.05),
    Dense(units=5, activation='linear', activity_regularizer=regularizers.l1(10e-5))
])

stacked_decoder = Sequential([
    Dropout(0.05),
    Dense(units=29, activation='linear')
])

stacked_ae = Sequential([stacked_encoder,
                         stacked_decoder])
stacked_ae.compile(optimizer='adam',
                   loss='mean_squared_error',
                   metrics=['accuracy'])

# fitting the model
history = stacked_ae.fit(x=X_train_AE, y=X_train_AE,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_train_AE, X_train_AE),
                    verbose=1)

# evaluating the model
predictions = stacked_ae.predict(X_test, verbose=1)
anomaly_scores_ae = anomaly_scores(X_test, predictions)
prediction_df = plot_results(y_test, anomaly_scores_ae, True)

#
# temp_df = pd.DataFrame(stacked_encoder.predict(x=X_train_AE))
# temp_df.to_csv('5d.csv')