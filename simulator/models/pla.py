from simulator.models.algorithms import PredictionModel
import numpy as np
from collections import deque


class PLA(PredictionModel):

    def __init__(self, k):
        super().__init__(model_type="pla")
        self.__k = k
        self.check_begin = self.__k
        self.queue_size = self.__k
        self.data = deque(maxlen=self.queue_size)
        self.__a = 0
        self.__b = 0

    def __best_fit_line(self, t):
        x = np.array(range(t - self.__k + 1, t + 1))
        y = np.array(self.data)
        cov = np.cov(x, y)[0, 1]
        var = np.var(x)
        self.__a = cov / var
        self.__b = np.mean(y) - self.__a * np.mean(x)

    def predict(self, t, **kwargs):
        output = self.__a * t + self.__b
        return output

    def update(self, t, cache):
        for element in cache:
            self.data.append(element)

        if t >= self.check_begin:
            self.__best_fit_line(t)
        return
