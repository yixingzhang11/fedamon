from statsmodels.tsa.ar_model import AutoReg
from collections import deque
from simulator.models.algorithms import PredictionModel
import numpy as np
from simulator.config_loader import *


class AutoRegModel(PredictionModel):

    def __init__(self, lags, sample):
        super().__init__(model_type="ar")
        self.__lags = lags
        self.queue_size = int(sample)
        self.data = deque(maxlen=self.queue_size)
        self.last_fit_time = None
        self.check_begin = 2 * self.__lags + 1
        self.regResult = None
        self.predict_begin_time = 0

    def __fit_model(self, t):
        self.regResult = AutoReg(np.array(self.data), lags=self.__lags).fit()
        self.last_fit_time = t
        return self.regResult

    def predict(self, t, **kwargs):
        output = self.regResult.predict(
            start=t - (self.last_fit_time - self.predict_begin_time + 1),
            end=t - (self.last_fit_time - self.predict_begin_time + 1),
        )  # 左闭右闭
        return output

    def update(self, t, cache):
        self.updated = cache[-1]

        for element in cache:
            self.data.append(element)

        if t >= self.check_begin:
            self.__fit_model(t)
            self.predict_begin_time = len(self.data)
        return
