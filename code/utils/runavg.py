class RunningAverage:
    def __init__(self, window_size=100):
        self.value = None
        self.lambda_ = 2/(window_size + 1)
        self.history = []
        self.grad_computations_history = []

    def update(self, new_value, grad_computations):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.lambda_ * new_value + (1 - self.lambda_) * self.value
        self.history.append(self.value)
        self.grad_computations_history.append(grad_computations)

