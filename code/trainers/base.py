import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import json
import os
import random

from utils.runavg import RunningAverage

class BaseTrainer:
    def __init__(
        self,
        model,
        loss_fn,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs=10,
        device='cpu',
        lr=0.1
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
        self.lr = lr
        self.train_loss = RunningAverage(window_size=1)
        self.train_accuracy = RunningAverage(window_size=1)
        self.test_loss = RunningAverage(window_size=1)
        self.test_accuracy = RunningAverage(window_size=1)
        self.grad_computations = 0

    def train(self):

        for epoch in tqdm(range(self.epochs), desc=f"{self.__class__.__name__}", ncols=100):
            avg_loss, avg_acc = self._run_epoch(self.train_loader, epoch, is_train=True)
            self.train_loss.update(avg_loss, self.grad_computations)
            self.train_accuracy.update(avg_acc, self.grad_computations)

            avg_loss, avg_acc = self._run_epoch(self.test_loader, epoch, is_train=False)
            self.test_loss.update(avg_loss, self.grad_computations)
            self.test_accuracy.update(avg_acc, self.grad_computations)

            tqdm.write(
                f"Train Loss: {self.train_loss.value:.4f}, Train Acc: {self.train_accuracy.value:.4f} | "
                f"Test Loss: {self.test_loss.value:.4f}, Test Acc: {self.test_accuracy.value:.4f}"
            )

    def _get_epoch_batches(self, data_loader, is_train=True):
        if is_train and getattr(self, "shuffle_batches", False):
            batches = list(data_loader)
            random.shuffle(batches)
            return batches
        else:
            return data_loader

    def _run_epoch(self, data_loader, epoch, is_train=True):
        mode_name = "Train" if is_train else "Eval"
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.set_grad_enabled(is_train):
            for inputs, targets in tqdm(self._get_epoch_batches(data_loader, is_train), desc=mode_name, leave=False, ncols=100):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                loss, correct = self._run_batch(inputs, targets, is_train)
                total_loss += loss
                total_correct += correct
                total_samples += len(targets)

        avg_loss = total_loss / len(data_loader)
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        return avg_loss, avg_acc

    def _run_batch(self, inputs, targets, is_train=True):
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)

        if is_train:
            self._optimizer_step(loss)

        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()

        return loss.item(), correct

    def _optimizer_step(self, loss):
        raise NotImplementedError

    def dump_histories(self):
        os.makedirs("experiments", exist_ok=True)

        d = {
            "train_loss": self.train_loss.history,
            "train_acc": self.train_accuracy.history,
            "test_loss": self.test_loss.history,
            "test_acc": self.test_accuracy.history,
            "train_loss_grad_computations": self.train_loss.grad_computations_history,
            "train_acc_grad_computations": self.train_accuracy.grad_computations_history,
            "test_loss_grad_computations": self.test_loss.grad_computations_history,
            "test_acc_grad_computations": self.test_accuracy.grad_computations_history,
        }
        with open(f"experiments/{self.__class__.__name__[:-7]}_{self.lr}_{datetime.now().strftime('%m-%d_%H-%M')}.json", "w") as f:
            json.dump(d, f)
