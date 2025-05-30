from torch.optim import SGD
from trainers.base import BaseTrainer

class SGDTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = SGD(self.model.parameters(), lr=self.lr)

    def _optimizer_step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.grad_computations += 1
        self.optimizer.step()

