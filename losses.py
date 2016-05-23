import theano.tensor as T


def scale_invariant_error(predictions, targets):

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
    return (T.sum(n_valid * T.sum(d**2, axis=1)) - _lambda_ * T.sum(T.sum(d, axis=1)**2))/ T.maximum(T.sum(n_valid**2), 1)
