# -*- coding: utf-8 -*-
import numpy as np


def average_difference(data: np.ndarray) -> float:
    """计算平均偏差（不常用）

    :param data: 要计算的数据
    :return: 返回平均偏差

    Example:

    >>> average_difference(np.array([0.2,0.3,0.2,0.1]))
    0.05000000000000001
    """
    x_ = data.mean()
    return np.abs(data - x_).mean()


def std(data: np.ndarray, ddof: int = 1) -> float:
    """ 计算修正的样本标准差

    :param data: 要求标准差的数据
    :param ddof: 修正值，除法分母为n - ddof，默认为1
    :return: 标准差

    >>> std(np.array([0.2,0.2,0.2,0.2]))
    0.0
    """
    return data.std(ddof=ddof)


def variation_coefficient(data: np.ndarray, ddof: int = 1) -> float:
    """计算样本的变异系数

    :param data:
    :param ddof:
    :return:

    >>> variation_coefficient(np.array([41.24,41.27,41.23,41.26]))
    0.00044260408687289926
    """
    return std(data,ddof) / data.mean()
