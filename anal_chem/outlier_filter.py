# -*- coding: utf-8 -*-


from math import sqrt

# 提供可疑数据的取舍相关包
import numpy as np
import scipy.stats


def grubbs_test(data: np.ndarray, alpha: float = 0.05) -> (bool, bool):
    """使用Grubbs测试异常值

    :param data: 要进行双侧Grubbs检验的数据
    :param alpha: 显著性水平，默认取0.05
    :return: 按顺序返回最小值和最大值是否应该舍去。

    >>> grubbs_test(np.array([0.3,-10000,100000,0.3,0.3,0.3,0.35,0.33,0.35,0.32,0.32,0.32,0.3,0.3,0.3]))
    (False, True)

    Grubbs检验的是：

    .. math:: H_0:没有错误数据
    .. math:: H_a - 有且仅有一个错误数据

    一次检验只会决定是否去掉一个数据，这个例子虽然-10000也应该被判断为“异常值”，但是100000太大，检定法便认为应该先去除它。


    >>> grubbs_test(np.array([0.3,0.35,0.323,0.331,0.9]))
    (False, True)

    第一个False意为最小值正常，第二个True意为最大值异常

    >>> grubbs_test(np.array([0.33,0.33,0.33,0.33,0.33,0.33]))
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

    def _t_critical_sq(alpha, degrees_of_freedom):
        return scipy.stats.t.interval(1 - alpha, degrees_of_freedom)[1] ** 2

    def _G_critical(alpha, n):
        return (n - 1) / sqrt(n) * sqrt(_t_critical_sq(alpha / (2 * n), n - 2) /
                                        ((n - 2) + _t_critical_sq(alpha / (2 * n), n - 2)))

    x_avg = data.mean()
    s = data.std(ddof=1)
    if s < 1e-8:
        return False, False
    ma = max(*(data - x_avg))
    mi = min(*(data - x_avg))
    n = data.shape[0]
    return -mi >= s * _G_critical(alpha, n), ma >= s * _G_critical(alpha, n)


def grubbs_test_iter(data: np.ndarray, alpha: float = 0.05) -> (np.ndarray, np.ndarray):
    """通过迭代的方法删除异常值

    这个函数迭代调用grubbs_test函数，一次删除一个异常值。

    :param data: 要进行迭代grubbs_test的数据
    :param alpha: 显著性水平，通常取0.05
    :return: 返回两个ndarray，第一个数组为bool类型，代表结果中元素是否被删去，第二个为去除异常值后的数组

    >>> grubbs_test_iter(np.array([0.3,0.3,0.3,0.3,0.3,0.3,0.3,-100,100000]))
    (array([ True,  True,  True,  True,  True,  True,  True, False, False]), array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))

    >>> grubbs_test_iter(np.array([0.3,0.2,0.4,0.3,0.25,0.35,-100,50000,-3000]))[1]
    array([0.3 , 0.2 , 0.4 , 0.3 , 0.25, 0.35])

    本例中，grubbs检定法删去了末尾的几个离群值

    >>> grubbs_test_iter(np.array([0.3,0.2,0.4,0.3,0.25,0.35,-100,500000,100]))[1]
    array([   0.3 ,    0.2 ,    0.4 ,    0.3 ,    0.25,    0.35, -100.  ,
            100.  ])
    """

    def find_arg_with_flag(data: np.ndarray, flag: np.ndarray, check_func):
        arg = None
        cur = None
        for i, _ in enumerate(data):
            if flag[i]:
                if arg is None:
                    arg = i
                    cur = data[i]
                if check_func(data[i], cur):
                    cur = data[i]
                    arg = i
        return arg

    flag = np.zeros(data.shape[0]) == 0
    status = grubbs_test(data[flag], alpha)
    while status[0] or status[1]:
        status = grubbs_test(data[flag], alpha)
        if status[0]:
            # 删去最小值
            x = find_arg_with_flag(data, flag, lambda a, b: a < b)
            flag[x] = False
        if status[1]:
            x = find_arg_with_flag(data, flag, lambda a, b: a > b)
            flag[x] = False
    return flag, data[flag]
