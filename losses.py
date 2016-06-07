import theano.tensor as T
import numpy as np



def l2(predictions, targets):
    # Compute mask
    mask = T.gt(targets, 0.1)
    # Compute n of valid pixels
    n_valid = T.sum(mask)
    # Apply mask 
    d = (predictions - targets) * mask
    return np.sqrt(T.sum((d)**2) / n_valid)


def berhu(predictions, targets,s=0.2,e=0.01):
    # Compute mask
    mask = T.gt(targets, e)
    # Compute n of valid pixels
    n_valid = T.sum(mask)
    # Redundant mult here 
    r = (predictions - targets) * mask
    c = s * T.max(T.abs_(r))
    a_r = T.abs_(r)
    b = T.switch(T.lt(a_r, c), a_r, ((r**2) + (c**2))/(2*c))
    return T.sum(b)/n_valid

def mse(predictions, targets):
    # Compute mask
    mask = T.gt(targets, 0.1)
    # Compute n of valid pixels
    n_valid = T.sum(mask)
    # Apply mask 
    d = (predictions - targets) * mask
    return T.sum((d)**2) / n_valid


def tukey_biweight(predictions, targets, c=4.685, s=1.4826):
    """
    Tukey's biweight function expressed in theano as in
    :param predictions: Prediction tensor
    :param targets: Target tensor
    :param c: Tukey tuning constant
    :param s: Consistence scale parameter
    :return: Cost function
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
    m_t = T.switch(mask, target, 0)

    def median(tensor):
        """
        MAD tensor from https://groups.google.com/forum/#!topic/theano-users/I4eHjbAetEQ
        :param tensor: Input tensor
        :return: Median expression
        """
        tensor = tensor.flatten(1)
        return T.switch(T.eq((tensor.shape[0] % 2), 0),
                        # if even vector
                        T.mean(T.sort(tensor)[((tensor.shape[0] / 2) - 1): ((tensor.shape[0] / 2) + 1)]),
                        # if odd vector
                        T.sort(tensor)[tensor.shape[0] // 2])

    def mad(tensor):
        """
        Median absolute deviation
        :param tensor: Input tensor
        :return: MAD
        """
        med = median(tensor=tensor)
        return median(T.abs_(tensor - med))

    # Residual
    r_i = (m_pred - m_t)
    # r_i = r_i / (s * mad(r_i))
    r_i = r_i / r_i.std()
    # Compute the masking vectors
    tukey_mask = T.gt(T.abs_(r_i), c)

    # Cost
    cost = (c ** 2 / 6) * (1-(1 - (r_i / c) ** 2) ** 3)
    # Aggregate
    return T.sum(T.sum(T.switch(tukey_mask, (c ** 2) / 6., cost), axis=1)) / T.maximum((T.sum(n_valid)), 1)


def spatial_gradient(prediction, target, l=0.5):
    # Flatten input to make calc easier
    pred = prediction
    pred_v = pred.flatten(2)
    target_v = target.flatten(2)
    # Compute mask
    mask = T.gt(target_v,0.)
    # Compute n of valid pixels
    n_valid = T.sum(mask, axis=1)
    # Apply mask and log transform
    m_pred = pred_v * mask
    m_t = T.switch(mask, T.log(target_v),0.)
    d = m_pred - m_t

    # Define scale invariant cost
    scale_invariant_cost = (T.sum(n_valid * T.sum(d**2, axis=1)) - l*T.sum(T.sum(d, axis=1)**2))/ T.maximum(T.sum(n_valid**2), 1)

    # Add spatial gradient components from D. Eigen DNL

    # Squeeze in case
    if pred.ndim == 4:
        pred = pred[:,0,:,:]
    if target.ndim == 4:
        target = target[:,0,:,:]
    # Mask in tensor form
    mask_tensor = T.gt(target,0.)
    # Project into log space
    target = T.switch(mask_tensor, T.log(target),0.)
    # Stepsize
    h = 1
    # Compute spatial gradients symbolically
    p_di = (pred[:,h:,:] - pred[:,:-h,:]) * (1 / np.float32(h))
    p_dj = (pred[:,:,h:] - pred[:,:,:-h]) * (1 / np.float32(h))
    t_di = (target[:,h:,:] - target[:,:-h,:]) * (1 / np.float32(h))
    t_dj = (target[:,:,h:] - target[:,:,:-h]) * (1 / np.float32(h))
    m_di = T.and_(mask_tensor[:,h:,:], mask_tensor[:,:-h,:])
    m_dj = T.and_(mask_tensor[:,:,h:], mask_tensor[:,:,:-h])
    # Define spatial grad cost
    grad_cost = T.sum(m_di * (p_di - t_di)**2) / T.sum(m_di) + T.sum(m_dj * (p_dj - t_dj)**2) / T.sum(m_dj)
    # Compute final expression
    return scale_invariant_cost + grad_cost

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
