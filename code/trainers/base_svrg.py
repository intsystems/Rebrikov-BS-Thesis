# trainers/base_svrg.py
import torch
from tqdm import tqdm

from trainers.base import BaseTrainer

class BaseSVRGTrainer(BaseTrainer):
    def __init__(self, *args, update_freq_epochs=5, **kwargs):
        """
        :param update_freq_epochs: Для классического SVRG – частота обновления референсной точки (каждые k эпох).
        Остальные параметры наследуются от BaseTrainer.
        """
        super().__init__(*args, **kwargs)
        self.update_freq_epochs = update_freq_epochs
        self.ref_params = None  # Сохранённая (референсная) точка
        self.ref_grad = None    # Полный (усреднённый) градиент в точке ref_params
        self._last_inputs = None  # Сохраняем последний батч для вычисления градиента в ref-точке
        self._last_targets = None

    def _update_ref_and_mu(self):
        """
        Вычисление референсной точки и полного градиента (mu):
          1. Сохраняем текущие параметры модели в ref_params.
          2. Проходим по всему train_loader, аккумулируем градиенты и усредняем их.
        """
        self.ref_params = [p.detach().clone() for p in self.model.parameters()]

        # Обнуляем градиенты
        for p in self.model.parameters():
            p.grad = None

        total_batches = 0
        self.model.train()
        for inputs, targets in tqdm(self.train_loader, desc="Computing full gradient", leave=False, ncols=100):
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

        # Обнуляем градиенты снова, чтобы не мешали дальнейшим вычислениям
        for p in self.model.parameters():
            p.grad = None

    def _optimizer_step(self, loss):
        """
        Один шаг алгоритма SVRG:
          1. Вычисляем градиент в текущей точке (g_current) с сохранением вычислительного графа.
          2. Переключаем параметры на референсную точку и вычисляем градиент на том же батче (g_ref).
          3. Обновляем параметры по правилу: p = p - lr * (g_current - g_ref + ref_grad)
        """
        # Шаг 1. Вычисление градиента g_current
        for p in self.model.parameters():
            p.grad = None
        loss.backward(retain_graph=True)
        self.grad_computations += 1
        self.grads_current = [p.grad.detach().clone() for p in self.model.parameters()]

        # Шаг 2. Вычисление градиента g_ref на том же батче, но в точке ref_params
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

        # Восстанавливаем параметры текущей модели
        with torch.no_grad():
            for p, bkp in zip(self.model.parameters(), backup_params):
                p.copy_(bkp)

        # Шаг 3. Обновление параметров
        with torch.no_grad():
            for p, g_cur, g_ref, mu in zip(self.model.parameters(), self.grads_current, grads_ref, self.ref_grad):
                grad_svrg = g_cur - g_ref + mu
                p.add_(grad_svrg, alpha=-self.lr)

    def _run_batch(self, inputs, targets, is_train=True):
        self._last_inputs = inputs
        self._last_targets = targets
        return super()._run_batch(inputs, targets, is_train=is_train)
