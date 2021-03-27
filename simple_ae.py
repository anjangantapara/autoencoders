""" Code to create simple AE for TS"""
"""Main"""
import numpy as np
import pandas as pd
import os, time, re
import pickle, gzip

"""Data Viz"""
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import matplotlib as mpl

"""Data preparation and model evaluation"""
from sklearn import preprocessing as pp
from sklearn.model_selection import test_train_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve,auc, roc_auc_score

"""Algos"""
import lighgbm as lgb

"""TensorFlow and Keras"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Activation,Dropout, Dense
from tensorflow.keras.layers import BatchNormalization,Input, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras.losses import mse, binary_crossentropy

data = pd.read_csv("/home/anjan/work/datasets/creditcard.csv")
dataX = data.copy().drop(['Class','Time'],axis=1)
dataY = data['Class'].copy()
featuresToScale = data.columns
sX = pp.StandardScaler(copy=True, with_mean=True, with_std=True)
dataX.loc[:,featuresToScale] = sX.fit_transform(dataX[featuresToScale])

X_train, X_test, y_train, y_test =\
    test_train_split(dataX,dataY,test_size=0.33,\
                     radom_state=42, stratify=dataY)
X_train_AE = X_train.copy()
X_test_AE =  X_train_AE.copy()

def anamoly_score(originalDF, reducedDF):

