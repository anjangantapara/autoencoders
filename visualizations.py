from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, auc


def plot_results(true_labels: np.ndarray, anomaly_scores: np.ndarray, return_predictions: Optional[bool] = False) -> \
        Optional[pd.DataFrame]:
    """

    :param true_labels: real labels from the dataset
    :param anomaly_scores: predicted
    :param return_predictions: if True, returns the predictions_df
    :return:
    """
    predictions_df = pd.concat([true_labels, anomaly_scores], axis=1)
    predictions_df.columns = ['true_labels', 'anomaly_scores']
    precision, recall, thresholds = precision_recall_curve(predictions_df['true_labels'], predictions_df['anomaly_scores'])
    avg_precision = average_precision_score(predictions_df['true_labels'], predictions_df['anomaly_scores'])
    plt.step(recall, precision, color='k', alpha=0.7, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.1, color='k')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.title('Precision Recall curve: Average Precision = \
        {0:0.2f}'.format(avg_precision))
    fpr, tpr, thresholds = roc_curve(predictions_df['true_labels'],
                                     predictions_df['anomaly_scores'])
    area_under_roc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC: AUC= {0:0.2f}'.format(area_under_roc))
    plt.legend(loc="lower right")
    plt.show()
    if return_predictions:
        return predictions_df
