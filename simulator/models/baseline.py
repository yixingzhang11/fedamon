from simulator.models.algorithms import PredictionModel


class basic(PredictionModel):
    def __init__(self, initial_value):
        super().__init__(model_type="basic", initial_value=initial_value)
        self.check_begin = 1

    def predict(self, t, **kwargs):
        return self.updated

    def update(self, t, cache):
        self.updated = cache[-1]
        return


class simpleapprox(PredictionModel):
    def __init__(self, initial_value):
        super().__init__(model_type="sa", initial_value=initial_value)
        self.check_begin = 1

    def predict(self, t, **kwargs):
        return self.updated

    def update(self, t, cache):
        self.updated = cache[-1]
        return


class Naive(PredictionModel):
    def __init__(self, initial_value=None):
        super().__init__(model_type="naive", initial_value=initial_value)
        self.check_begin = 0

    def predict(self, t, **kwargs):
        return float("inf")

    def update(self, t, cache):
        self.updated = cache[-1]
        return
