# trainers/no_full_grad_svrg.py
import torch
from trainers.base_svrg import BaseSVRGTrainer

class NFGSVRGTrainer(BaseSVRGTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Отключаем вероятностное обновление ref (не нужно в данном алгоритме)
        # инициализируем переменные для накопления среднего градиента
        self.tilde_v = None    # running average градиентов (ṽ)
        self.shuffle_batches = True
        self.batch_counter = 0 # счётчик батчей в эпохе

    def _update_ref_and_mu(self):
        if self.ref_params is None:
            # Инициализация при первом вызове
            self.ref_params = [p.detach().clone() for p in self.model.parameters()]
            self.ref_grad = [torch.zeros_like(p) for p in self.model.parameters()]
            self.tilde_v = [torch.zeros_like(p) for p in self.model.parameters()]
            self.batch_counter = 0
            return
        self.ref_params = [p.detach().clone() for p in self.model.parameters()]
        self.ref_grad = [v.clone()/self.batch_counter for v in self.tilde_v]
        self.tilde_v = [torch.zeros_like(p) for p in self.model.parameters()]
        self.batch_counter = 0

    def _optimizer_step(self, loss):
        # Сначала вызываем базовую реализацию, которая рассчитает self.grads_current
        super()._optimizer_step(loss)
        
        for i in range(len(self.tilde_v)):
            self.tilde_v[i]  +=  self.grads_current[i]
        self.batch_counter += 128

    
    def _run_epoch(self, data_loader, epoch, is_train=True):
        if is_train:
            self._update_ref_and_mu()
        return super()._run_epoch(data_loader, epoch, is_train=is_train)


