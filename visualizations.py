from typing import Optional

import matplotlib.pyplot as plt
import numpy as  np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc


def plotResults(trueLabels: np.ndarray, anomalyScores: np.ndarray, returnPreds: Optional[bool] = False) -> Optional[
    pd.DataFrame]:
    """

    :param trueLabels: real labels from the dataset
    :param anomalyScores: predicted
    :param returnPreds: if True, returns the preds
    :return:
    """
    preds = pd.concat([trueLabels, anomalyScores], axis=1)
    preds.columns = ['trueLabels', 'anomalyScores']
    precision, recall, thresholds = precision_recall_curve(preds['trueLabels'], preds['anomalyScores'])
    avg_precision = average_precision_score(preds['trueLabels'], preds['anomalyScores'])
    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.1, color='k')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.title('Precision Recall curve: Average Precision = \
        {0:0.2f}'.format(avg_precision))
    fpr, tpr, thresholds = roc_curve(preds['trueLabels'], \
                                     preds['anomalyScores'])
    areaUderROC = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [1, 0], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: AUC= {0:0.2f}'.format(areaUderROC))
    plt.legend(loc="lower right")
    plt.show()
    if returnPreds:
        return preds
