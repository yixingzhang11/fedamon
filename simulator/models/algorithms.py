class PredictionModel:
    def __init__(self, model_type, initial_value=None):
        self.model_type = model_type
        self.check_begin = None
        self.updated = initial_value
        self.output = None

    def predict(self, t, **kwargs):
        raise NotImplementedError("Define in subclass of PredictionModel")

    def update(self, t, cache):
        raise NotImplementedError("Define in subclass of PredictionModel")
