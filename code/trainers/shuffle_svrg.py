# trainers/shuffle_svrg.py
import random
from trainers.base_svrg import BaseSVRGTrainer
from dataloader.cifar import NUM_OF_BATCHES  # NUM_OF_BATCHES – общее число батчей в эпохе

class ShuffleSVRGTrainer(BaseSVRGTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Включаем режим перемешивания батчей
        self.shuffle_batches = True
        # Вычисляем вероятность обновления ref-состояния:
        self.p = 1 / (self.update_freq_epochs * NUM_OF_BATCHES)

    def _optimizer_step(self, loss):
        if self.ref_params is None:
            self._update_ref_and_mu()
        # Вызываем стандартный шаг из BaseSVRGTrainer
        super()._optimizer_step(loss)
        # С вероятностью p обновляем ref-состояние
        if random.random() < self.p:
            self._update_ref_and_mu()