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
    loss = np.sum((np.array(originalDF) - \
                   np.array(reducedDF) )**2.0 ,axis=1)
    loss = pd.Series(data=loss,index=originalDF.index)
    loss = (loss - np.min(loss))/(np.max(loss)-np.min(loss))
    return(loss)

def plotResults(trueLabels, anomalyScores, returnPreds = False):
    preds = pd.concat(trueLabels,anomalyScores, axis=1)
    preds.columns = ['trueLabels', 'anomalyScores']
    precision, recall, thresholds = precision_recall_curve(preds['trueLabels'], preds['anomalyScores'])
    avg_precision = average_precision_score(preds['trueLabels'],preds['anomalyScores'])
    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.refill_between(recall,precision,step='post',alpha=0.1,color='k')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.title('Precision Recall curve: Average Precision = \
        {0:0.2f}'.format(avg_precision))
    fpr,tpr=thresholds=roc_curve( preds['trueLabels'],\
                        preds['anamolyScores'])
    areaUderROC = auc(fpr,tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, labels='ROC curve')
    plt.plot([0,1],[1,0],color='k',lw=2,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: AUC= {0:0.2f}'.format(areaUderROC))
    plt.legend(loc="lowe right")
    plt.show()






