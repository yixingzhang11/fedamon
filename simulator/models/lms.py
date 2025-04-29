import numpy as np
from simulator.models.algorithms import PredictionModel
from collections import deque
from simulator.config_loader import *


class LMSFilter(PredictionModel):
    def __init__(self, k, w, lr):
        super().__init__(model_type="lms")
        self.__k = k
        self.__learning_rate = 0
        self.__w = self.__init_weights(w, self.__k)
        self.check_begin = self.__k
        self.queue_size = self.__k
        self.data = deque(maxlen=self.queue_size)
        self.v_hat = 0
        self.__lr = lr

    def __init_weights(self, w, __k):
        if isinstance(w, str):
            if w == "random":
                w = np.random.normal(0, 0.5, __k)
            elif w == "zeros":
                w = np.zeros(__k)
            else:
                raise ValueError("Impossible to understand the w")
        return w

    def __learning_rate_calculation(self):
        denominator = np.sum(np.array(self.data) ** 2)
        if denominator == 0:
            self.__learning_rate = self.__learning_rate
        else:
            upper_bound = 1 / (np.sum(np.array(self.data) ** 2) / np.array(self.data).size)
            self.__learning_rate = upper_bound / self.__lr

    def predict(self, last_prediction, **kwargs):
        if last_prediction:
            self.data.append(last_prediction)
        self.v_hat = np.dot(self.__w, np.array(self.data))
        if np.isnan(self.v_hat):
            self.v_hat = 0
        return self.v_hat

    def __weight_update(self):
        e = self.data[-1] - self.v_hat
        self.__learning_rate_calculation()
        self.__w += self.__learning_rate * np.array(self.data) * e
        self.__w[np.isnan(self.__w)] = 0

    def update(self, t, cache):
        for element in cache:
            self.data.append(element)

        if t >= self.check_begin:
            self.__weight_update()

        return
