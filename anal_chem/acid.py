# -*- coding: utf-8 -*-
import numpy as np


class Acid:
    def __init__(self, pk=True, *args):
        self.K = np.array(args)
        if pk:
            self.K = 10 ** -self.K

    def delta(self, c_H):
        pass
