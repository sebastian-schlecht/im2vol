import theano.tensor as T
import numpy as np


def mad(arr):
    """
    Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variabililty of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    arr = np.ma.array(arr).compressed()  # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def tukey_biweight(predictions, targets, c=4.685, s=1.4826):
    """
    Tukey's biweight function expressed in theano as in
    :param predictions: Prediction tensor
    :param targets: Target tensor
    :param c: Tukey tuning constant
    :param s: Consistence scale parameter
    :return:
    """
    # Flatten input to make calc easier
    pred = predictions.flatten(2)
    target = targets.flatten(2)
    # Compute mask
    mask = T.gt(target, 0)
    # Compute n of valid pixels
    n_valid = T.sum(mask, axis=1)
    # Apply mask and log transform
    m_pred = pred * mask
    m_t = T.switch(mask, T.log(target), 0)

    def median(tensor):
        """
        MAD tensor from https://groups.google.com/forum/#!topic/theano-users/I4eHjbAetEQ
        :param tensor: Input tensor
        :return: MAD expression
        """
        return T.switch(T.eq((tensor.shape[0] % 2), 0),
                        # if even vector
                        T.mean(T.sort(tensor)[((tensor.shape[0] / 2) - 1): ((tensor.shape[0] / 2) + 1)]),
                        # if odd vector
                        T.sort(tensor)[tensor.shape[0] // 2])

    def mad(tensor):
        """
        Median absolute deviation
        :param tensor: Input tensor
        :return:
        """
        med = median(tensor=tensor)
        return median(T.abs_(tensor - med))

    # Residual
    r_i = (m_pred - m_t) / s * mad(m_t)
    # Compute the masking vectors
    tukey_mask = T.gt(r_i, c)

    # Cost
    cost = (c ** 2 / 6) * (1 - (r_i / c) ** 2) ** 2
    return T.sum(T.sum(T.switch(tukey_mask, (c ** 2) / 6, cost), axis=1)) / T.max(n_valid, 1)


def scale_invariant_error(predictions, targets):
    """
    Scale invariant error in log space
    :param predictions: Prediction tensor
    :param targets: Target tensor
    :return: theano expression
    """
    _lambda_ = 0.5

    # Flatten input to make calc easier
    pred = predictions.flatten(2)
    target = targets.flatten(2)
    # Compute mask
    mask = T.gt(target, 0)
    # Compute n of valid pixels
    n_valid = T.sum(mask, axis=1)
    # Apply mask and log transform
    m_pred = pred * mask
    m_t = T.switch(mask, T.log(target), 0)
    d = m_pred - m_t

    # Define cost
    return (T.sum(n_valid * T.sum(d ** 2, axis=1)) - _lambda_ * T.sum(T.sum(d, axis=1) ** 2)) / T.maximum(
        T.sum(n_valid ** 2), 1)
