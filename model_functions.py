import numpy as np
import pandas as pd


def anomalyScores(originalDF: pd.DataFrame, reducedDF: pd.DataFrame) -> np.ndarray:
    """

    :param originalDF: original data frame
    :param reducedDF: predicted data frame
    :return: loss of each element in a numpy array. Index of the loss same as original df
    """
    loss = np.sum((np.array(originalDF) -
                   np.array(reducedDF)) ** 2.0, axis=1)
    loss = pd.Series(data=loss, index=originalDF.index)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))
    return (loss)
