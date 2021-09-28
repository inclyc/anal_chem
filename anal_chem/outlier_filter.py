# -*- coding: utf-8 -*-


from math import sqrt

# 提供可疑数据的取舍相关包
import numpy as np
import scipy.stats


def _t_critical_sq(alpha, degrees_of_freedom):
    return scipy.stats.t.interval(1 - alpha, degrees_of_freedom)[1] ** 2


def _G_critical(alpha, n):
    return (n - 1) / sqrt(n) * sqrt(_t_critical_sq(alpha / (2 * n), n - 2) /
                                    ((n - 2) + _t_critical_sq(alpha / (2 * n), n - 2)))


def grubbs_test(data: np.ndarray, alpha: float = 0.05) -> (bool, bool):
    """
    :param data: 要进行双侧Grubbs检验的数据
    :param alpha: 显著性水平，默认取0.05
    :return: 按顺序返回最小值和最大值是否应该舍去。
    >>> grubbs_test(np.array([0.3,-10000,100000,0.3,0.3,0.3,0.35,0.33,0.35,0.32,0.32,0.32,0.3,0.3,0.3]))
    (False, True)

    Grubbs检验的是
    H0 - 没有错误数据
    H1 - 有且仅有一个错误数据
    一次检验只会决定是否去掉一个数据，这个例子虽然-10000也应该被判断为“异常值”，但是100000太大，检定法便认为应该先去除它。


    >>> grubbs_test(np.array([0.3,0.35,0.323,0.331,0.9]))
    (False, True)

    第一个False意为最小值正常，第二个True意为最大值异常

    >>> grubbs_test(np.array([0.33,0.33,0.33,0.33,0.329,0.331]))
    (False, False)

    这个样本总体方差比较小，数据不接近正态分布

    >>> p = scipy.stats.norm.rvs(size=10000)
    >>> grubbs_test(p)
    (False, False)

    >>> grubbs_test(np.array([0.32,0.32,0.323,0.331,-0.9]))
    (True, False)

    >>> grubbs_test(np.array([1.25,1.27,1.31,1.40]))
    (False, False)

    >>> grubbs_test(np.array([0.3,0.3,0.3,0.3,0.3,0.3,0.3,-100]))
    (True, False)

    """
    x_avg = data.mean()
    s = data.std(ddof=1)
    ma = max(*(data - x_avg))
    mi = min(*(data - x_avg))
    n = data.shape[0]
    return -mi / s >= _G_critical(alpha, n), ma / s >= _G_critical(alpha, n)


def grubbs_test_iter(data: np.ndarray, alpha: float = 0.05) -> (np.ndarray, np.ndarray):
    """

    :param data:
    :param alpha:
    :return:

    >>> grubbs_test_iter(np.array([0.3,0.3,0.3,0.3,0.3,0.3,0.3,-100,100000]))
    (array([ True,  True,  True,  True,  True,  True,  True, False, False]), array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
    """
    flag = np.zeros(data.shape[0]) == 0
    while True:
        status = grubbs_test(data[flag], alpha)
        checked = False
        if status[0]:
            # 删去最小值
            x = data[flag].argmin()
            flag[x] = False
            checked = True
        if status[1]:
            x = data[flag].argmax()
            flag[x] = False
            checked = True
        if not checked:
            break
    return flag, data[flag]
