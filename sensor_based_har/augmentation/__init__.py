# TODO: sine (freq, amp)  # 元論文だと freq ratio という表記でランダムに選択されている
# TODO: square_pulse (freq, amp)
# TODO: sine_partial (ratio, freq)
# TODO: square_pulse_partial (ratio, freq)
# TODO: gaussian_noise_partial (ratio, sigma)
# TODO: FIR_lowpass (freq)
import random
import numpy as np
from scipy.interpolate import CubicSpline
import torch.nn.functional as F


__all__ = [
    'jitter', 'scaling', 'magnitude_warping', 'time_warping', 'rotation', 'permutation',
    'dropout', 'flipping', 'cutout', 'shift', 'window_warping', 'window_slicing',
    'mixup', 'cutmix',
]


def jitter(x, sigma=0.05):
    """ jitter or gaussian noise
    paper: https://arxiv.org/pdf/1706.00527.pdf

    ```math
    x = x + \mathcal{N}(0, \sigma)
    ```

    x.shape = (channel, length)
    """
    noise = np.random.normal(0, sigma, size=x.shape)
    return x + noise


def scaling(x, sigma=0.1):
    """ scaling
    paper: https://arxiv.org/pdf/1706.00527.pdf

    ```math
    s \sim \mathcal{N}(1, \sigma)
    x = sx
    ```

    x.shape = (channel, length)
    """
    factor = np.random.normal(1.0, sigma, size=(x.shape[0]))
    return np.multiply(x, factor[:, np.newaxis])


def magnitude_warping(x, sigma=0.2, knot=4):
    """ magnitude warping
    from: https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py#L67
    > Based on: T. T. Um et al, "Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks," in ACM ICMI, pp. 216-220, 2017.

    x.shape = (channel, length)
    """
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot + 2)
    )
    warp_steps = (
        np.ones((x.shape[0], 1)) * (np.linspace(0, x.shape[1]-1., num=knot+2))
    ).T
    ret = np.zeros_like(x)
    warper = np.array([
        CubicSpline(warp_steps[:, dim], random_warps[dim, :])(orig_steps)
        for dim in range(x.shape[0])
    ])
    ret = x * warper
    return ret


def time_warping(x, sigma=0.2, knot=4):
    """ time warping
    from: https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py#L67
    > Based on: T. T. Um et al, "Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks," in ACM ICMI, pp. 216-220, 2017.

    x.shape = (channel, length)
    """
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(
        loc=1.0, scale=sigma, size=(x.shape[0], knot+2)
    )
    warp_steps = (
        np.ones((x.shape[0], 1))*(np.linspace(0, x.shape[1]-1., num=knot+2))
    ).T

    ret = np.zeros_like(x)
    for dim in range(x.shape[0]):
        time_warp = CubicSpline(
            warp_steps[:,dim], warp_steps[:,dim] * random_warps[dim,:]
        )(orig_steps)
        scale = (x.shape[1]-1) / time_warp[-1]
        ret[dim,:] = np.interp(
            orig_steps,
            np.clip(scale * time_warp, 0, x.shape[1] - 1),
            x[dim,:]
        ).T
    return ret


def rotation(x):
    """ rotation
    from: https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py#L67

    x.shape = (channel, length)
    """
    flip = np.random.choice([-1, 1], size=(x.shape[0]))
    rotate_axis = np.arange(x.shape[0])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis] * x[rotate_axis, :]


def permutation(x, max_segments=4):
    """ permutation
    from: https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py#L67

    x.shape = (channel, length)
    """
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments)

    if num_segs > 1:
        splits = np.array_split(orig_steps, num_segs)
        random.shuffle(splits)
        warp = np.concatenate(splits).ravel()
        ret = x[:, warp]
    else:
        ret = x
    return ret


def dropout(x, p=0.1, mask_value=0):
    """ dropout

    based on: https://www.jstage.jst.go.jp/article/pjsai/JSAI2021/0/JSAI2021_2N1IS2a02/_pdf/-char/ja

    x.shape = (channel, length)
    """
    x = x.copy()
    mask = np.random.rand(*x.shape) < p
    x[mask] = mask_value
    return x


def flipping(x, p=0.5):
    """ flipping

    based on: https://www.jstage.jst.go.jp/article/pjsai/JSAI2021/0/JSAI2021_2N1IS2a02/_pdf/-char/ja

    x.shape = (channel, length)
    """
    n_features = x.shape[0]
    is_flipped = np.random.rand(n_features) < p
    flip = np.ones(n_features)
    flip[is_flipped] = -1
    flip = flip.reshape(-1, 1)
    return x * flip


def cutout(x, ratio=0.2, mask_value=0):
    """
    (この論文)[https://www.jstage.jst.go.jp/article/pjsai/JSAI2021/0/JSAI2021_2N1IS2a02/_pdf/-char/ja] の Description に沿った実装.

    x.shape = (channel, length)
    """
    x = x.copy()
    cut_window = int(x.shape[1] * ratio)
    cut_left = np.random.randint(x.shape[1] - cut_window)
    cut_right = cut_left + cut_window
    x[:, cut_left:cut_right] = mask_value
    return x


def shift(x, ratio=0.2, padding_mode='constant'):
    """
    (この論文)[https://www.jstage.jst.go.jp/article/pjsai/JSAI2021/0/JSAI2021_2N1IS2a02/_pdf/-char/ja] の Description に沿った実装.

    x.shape = (channel, length)
    """
    shift_width = int(x.shape[1] * ratio)
    dir = 1 if np.random.rand() < 0.5 else -1
    if dir > 0:
        pad_width = ((0, 0), (shift_width, 0))
    else:
        pad_width = ((0, 0), (0, shift_width))
    padded_x = np.pad(x, pad_width, mode=padding_mode)
    x = np.roll(padded_x, shift_width * dir, axis=1)
    if dir > 0:
        x = x[:, shift_width:].copy()
    else:
        x = x[:, :-shift_width].copy()
    return x


def window_warping(x, ratio: float=0.2, scale: float=0.25):
    """
    from: https://github.com/AlexanderVNikitin/tsgm/blob/main/tsgm/models/augmentations.py#L300

    Parameters
    ----------
    x : np.ndarray
        Input data tensor of shape (n_features, n_timesteps).
    ratio : float, optional
        The ratio of the window size relative to the total number of timesteps, by default 0.2
    scale : float, optional
        A scale for warping, by default 0.25
    """
    n_features = x.shape[0]
    n_timesteps = x.shape[1]

    warp_size = max(np.round(ratio * n_timesteps).astype(np.int64), 1)

    result = np.zeros((n_features, n_timesteps))
    window_starts = np.random.randint(low=0, high=n_timesteps - warp_size)
    window_ends = window_starts + warp_size

    for dim in range(n_features):
        start_seg = x[dim, :window_starts]
        warp_ts_size = max(round(warp_size * scale), 1)
        window_seg = np.interp(
            x=np.linspace(0, warp_size - 1, num=warp_ts_size),
            xp=np.arange(warp_size),
            fp=x[dim, window_starts:window_ends],
        )
        end_seg = x[dim, window_ends:]
        warped = np.concatenate((start_seg, window_seg, end_seg))
        result[dim, :] = np.interp(
            x=np.arange(n_timesteps),
            xp=np.linspace(0, n_timesteps - 1.0, num=warped.size),
            fp=warped,
        )
    return result


def window_slicing(x, ratio: float=0.9):
    """ window slicing

    from: https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py#L67
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    > Based on: A. Le Guennec, S. Malinowski, R. Tavenard, "Data Augmentation for Time Series Classification using Convolutional Neural Networks," in ECML/PKDD Workshop on Advanced Analytics and Learning on Temporal Data, 2016.

    x.shape = (channel, length)
    """
    target_len = int(x.shape[1] * ratio)
    if target_len >= x.shape[1]:
        return x
    start = np.random.randint(0, x.shape[1] - target_len, size=x.shape[0])
    end = (target_len + start)

    ret = np.zeros_like(x)
    for dim in range(x.shape[0]):
        ret[dim, :] = np.interp(
            np.linspace(0, target_len, num=x.shape[1]),
            np.arange(target_len),
            x[dim, start[dim]:end[dim]]
        ).T
    return ret


def mixup(x, y, num_classes, alpha=1.0):
    """mixup
    x.shape = (batch, channel, length)
    y.shape = (batch,) or (batch, num_classes)

    Returns
    -------
    mixed_x: torch.Tensor
        shape = (batch, channel, length)
    mixed_y: torch.Tensor
        shape = (batch, num_classes)
    """
    batch_size = x.shape[0]
    if len(y.shape) == 1:
        y = F.one_hot(y, num_classes)
    indices = np.random.permutation(batch_size)
    l = np.random.beta(alpha, alpha)
    l = np.maximum(l, 1-l)
    x_a, x_b = x, x[indices]
    y_a, y_b = y, y[indices]
    mixed_x = l * x_a + (1 - l) * x_b
    mixed_y = l * y_a + (1 - l) * y_b
    return mixed_x, mixed_y


def cutmix(x, y, num_classes, alpha=1.0):
    """cutmix
    x.shape = (batch, channel, length)
    y.shape = (batch,) or (batch, num_classes)

    Returns
    -------
    mixed_x: torch.Tensor
        shape = (batch, channel, length)
    mixed_y: torch.Tensor
        shape = (batch, num_classes)
    """
    batch_size = x.shape[0]
    if len(y.shape) == 1:
        y = F.one_hot(y, num_classes)
    indices = np.random.permutation(batch_size)
    cut_rate = np.random.beta(alpha, alpha)
    cut_rate = np.maximum(cut_rate, 1 - cut_rate)
    cut_window = int(x.shape[2] * cut_rate)
    cut_left = np.random.randint(x.shape[2] - cut_window)
    cut_right = cut_left + cut_window
    x_a, x_b = x, x[indices]
    y_a, y_b = y, y[indices]
    mixed_x = x_b.clone()
    mixed_x[:, :, cut_left:cut_right] = x_a[:, :, cut_left:cut_right]
    mixed_y = cut_rate * y_a + (1 - cut_rate) * y_b
    return mixed_x, mixed_y
