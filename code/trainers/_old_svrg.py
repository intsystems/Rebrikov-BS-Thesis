import torch
from tqdm import tqdm
import copy

from trainers.base import BaseTrainer

class SVRGTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_params = None
        self.ref_grad = None
        self.update_freq_epochs = 5

        self._last_inputs = None
        self._last_targets = None

    def _run_epoch(self, data_loader, epoch, is_train=True):
        if is_train and epoch % self.update_freq_epochs == 0:
            self._update_ref_and_mu()
        return super()._run_epoch(data_loader, epoch, is_train=is_train)

    def _run_batch(self, inputs, targets, is_train=True):
        self._last_inputs = inputs
        self._last_targets = targets
        return super()._run_batch(inputs, targets, is_train=is_train)

    def _update_ref_and_mu(self):
        self.ref_params = [p.detach().clone() for p in self.model.parameters()]

        for p in self.model.parameters():
            p.grad = None

        total_batches = 0
        self.model.train()  
        for inputs, targets in tqdm(self.train_loader, desc="Computing mu", leave=False, ncols=100):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.grad_computations += 1  
            total_batches += 1

        self.ref_grad = []
        with torch.no_grad():
            for p in self.model.parameters():
                p.grad /= total_batches
                self.ref_grad.append(p.grad.clone())

        for p in self.model.parameters():
            p.grad = None

    def _optimizer_step(self, loss):
        for p in self.model.parameters():
            p.grad = None
        loss.backward(retain_graph=True)
        self.grad_computations += 1
        grads_current = [p.grad.detach().clone() for p in self.model.parameters()]

        backup_params = [p.detach().clone() for p in self.model.parameters()]
        with torch.no_grad():
            for p, p_ref in zip(self.model.parameters(), self.ref_params):
                p.copy_(p_ref)

        for p in self.model.parameters():
            p.grad = None
        outputs_ref = self.model(self._last_inputs)
        loss_ref = self.loss_fn(outputs_ref, self._last_targets)
        loss_ref.backward()
        self.grad_computations += 1
        grads_ref = [p.grad.detach().clone() for p in self.model.parameters()]

        with torch.no_grad():
            for p, bkp in zip(self.model.parameters(), backup_params):
                p.copy_(bkp)

        with torch.no_grad():
            for p, g_cur, g_ref, mu in zip(self.model.parameters(), grads_current, grads_ref, self.ref_grad):
                grad_svrg = g_cur - g_ref + mu
                p.add_(grad_svrg, alpha=-self.lr)
