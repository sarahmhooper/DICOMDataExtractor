import numpy as np


def tn_scorer(golds, probs, preds, uids=None):
    """Accuracy classification score.

    :param golds: Ground truth (correct) target values.
    :type golds: 1-d np.array
    :param probs: Predicted target probabilities. (Not used!)
    :type probs: k-d np.array
    :param preds: Predicted target values.
    :type preds: 1-d np.array
    :param uids: Unique ids.
    :type uids: list
    :param normalize: Normalize the results or not, defaults to True
    :param normalize: bool, optional
    :return: Accuracy, if normalize == True, return the fraction of correctly
        predicted samples (float), else returns the number of correctly predicted
        samples (int).
    :rtype: dict
    """

    return {"tn": np.where(np.logical_and(golds == preds, golds == 2))[0].shape[0]}
   
