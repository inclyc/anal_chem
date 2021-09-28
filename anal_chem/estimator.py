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
    :return:
    """
    s = arr.std(ddof=ddof)
    x_avg = arr.mean()
    degree_free = arr.shape[0] - ddof
    n = arr.shape[0]
    t = np.array(scipy.stats.t.interval(alpha, degree_free))
    return x_avg + t * s / sqrt(n)
