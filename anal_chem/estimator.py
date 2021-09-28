# -*- coding: utf-8 -*-
from math import sqrt

import numpy as np
import scipy.stats


def sigma_estimator(arr: np.ndarray, ddof: int = 1, alpha: float = 0.95):
    """利用t分布对中心 sig 进行区间估计

    :param arr: 要进行区间估计的数据
    :param axis: 进行统计的轴（如果有多维疏数组）
    :param ddof: 自由度修正量
    :param alpha: [0,1]之间的数，置信度
    :return: np.ndarray 一个置信区间。

    Examples:

    >>> sigma_estimator(np.array([0.3,0.3,0.3,0.2,0.5,0.34]))
    array([0.22015365, 0.42651301])


    >>> sigma_estimator(np.array([0.3,0.3,0.3,0.2,0.5,0.34]),alpha=0.99)
    array([0.16148856, 0.4851781 ])

    本例选取了一个置信度更高的置信区间，因此区间更大了。
    """
    s = arr.std(ddof=ddof)
    x_avg = arr.mean()
    degree_free = arr.shape[0] - ddof
    n = arr.shape[0]
    t = np.array(scipy.stats.t.interval(alpha, degree_free))
    return x_avg + t * s / sqrt(n)
