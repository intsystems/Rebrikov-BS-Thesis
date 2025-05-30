import matplotlib.pyplot as plt
from datetime import datetime
import json
import numpy as np
import os

from dataloader.cifar import NUM_OF_BATCHES 

def rescale(arr):
    return (np.array(arr)/NUM_OF_BATCHES).tolist()

class Plotter:
    def __init__(self):
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.path = 'experiments'

    def plot_from_files(self, filenames):
        train_loss = {}
        train_acc = {}
        test_loss = {}
        test_acc = {}
        for filename, name in filenames:
            with open(f"{self.path}/{filename}", "r") as f:
                d = json.load(f)
                train_loss[name] = (d["train_loss"], d["train_loss_grad_computations"])
                train_acc[name] = (d["train_acc"], d["train_acc_grad_computations"])
                test_loss[name] = (d["test_loss"], d["test_loss_grad_computations"])
                test_acc[name] = (d["test_acc"], d["test_acc_grad_computations"])
        title = " ".join([fn[1] for fn in filenames])
        self.plot_loss_and_accuracy(title, train_loss, train_acc, test_loss, test_acc)

    def plot_loss_and_accuracy(self, title, train_loss, train_acc, test_loss, test_acc):
        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        self.plot_values(ax[0], train_loss, test_loss, "Loss")
        self.plot_values(ax[1], train_acc, test_acc, "Accuracy")
        fig.suptitle(title)
        plt.savefig(f"plots/{title}_{datetime.now().strftime('%m-%d_%H-%M')}.png")

    def plot_values(self, ax, train_values, test_values, name):
        for idx, (_, values) in enumerate(train_values.items()):
            ax.plot(rescale(values[1]), values[0], linestyle="--", linewidth=1, color=self.colors[idx])
        for idx, (_, values) in enumerate(test_values.items()):
            ax.plot(rescale(values[1]), values[0], linestyle="-", linewidth=2, color=self.colors[idx] )
        ax.set_xlabel("#epochs grads")
        ax.set_ylabel(name)
        ax.grid()
        handles = [plt.Line2D([0], [0], color=self.colors[idx], lw=2) for idx, (label, _) in enumerate(train_values.items())] + [plt.Line2D([0], [0], color="black", lw=2), plt.Line2D([0], [0], color="black", lw=1, linestyle="--")]
        ax.legend(handles, [f"{label}" for label in train_values.keys()] + ["Test", "Train"]) 
        ax.set_title(name)

    def find_latest(self, name):
        files = os.listdir(self.path)
        relevant_files = [f for f in files if f.startswith(name)]
        
        # Если подходящих файлов нет, выводим сообщение и выходим
        if not relevant_files:
            print(f"No files starting with '{name}' found")
            return

        return max(relevant_files, key=lambda f: f.split("_")[-2:])
        
        

    def plot_latest(self, names):
        files = map(self.find_latest, names)
        self.plot_from_files(list(zip(files, names)))


#test

# data_test_loss = {
#     "SGD": ([0.45, 0.35, 0.30, 0.25, 0.20], [1, 2, 3, 4, 5]),
#     "SVRG": ([0.50, 0.40, 0.35, 0.30, 0.25], [1, 2, 3, 4, 5]),
# }

# data_test_acc = {
#     "SGD": ([0.60, 0.65, 0.70, 0.75, 0.80], [1, 2, 3, 4, 5]),
#     "SVRG": ([0.55, 0.60, 0.65, 0.70, 0.75], [1, 2, 3, 4, 5]),
# }
# data_train_loss = {
#     "SGD": ([0.40, 0.30, 0.25, 0.20, 0.15], [1, 2, 3, 4, 5]),
#     "SVRG": ([0.45, 0.35, 0.30, 0.25, 0.20], [1, 2, 3, 4, 5]),
# }
# data_train_acc = {
#     "SGD": ([0.65, 0.70, 0.75, 0.80, 0.85], [1, 2, 3, 4, 5]),
#     "SVRG": ([0.60, 0.65, 0.70, 0.75, 0.80], [1, 2, 3, 4, 5]),
# }

# plotter = Plotter()
# plotter.plot_loss_and_accuracy("Test", data_test_loss, data_test_acc, data_train_loss, data_train_acc)   