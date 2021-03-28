import numpy as np
import pandas as pd


def anomaly_scores(original_df: pd.DataFrame, reduced_df: pd.DataFrame) -> np.ndarray:
    """
    Computes the anomaly scores. Consumes original df and reduced df.
    :param original_df: original data frame
    :param reduced_df: predicted data frame
    :return: loss of each ts row (element) is a numpy array. Index of the loss same as original df
    """
    loss = np.sum((np.array(original_df) -
                   np.array(reduced_df)) ** 2.0, axis=1)
    loss = pd.Series(data=loss, index=original_df.index)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
    return loss
