import torch 
from models.cifarnet import CifarNet
from dataloader.cifar import get_cifar10_dataloader
import numpy as np
import random

if __name__ == '__main__':
    LR = 0.01
    SGD_EPOCHS = 40
    FREQ_FG = 1/3

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    train_loader, test_loader = get_cifar10_dataloader(batch_size=128, num_workers=4)


    # DEVICE = 'cuda:3'
    DEVICE = 'mps'
    # print(f"[INFO] Using device: {DEVICE}, {torch.cuda.get_device_name(DEVICE)}")

    model = CifarNet(num_classes=10).to(DEVICE)

    from torch import nn
    loss_fn = nn.CrossEntropyLoss()

    # from trainers.sgd import SGDTrainer
    # trainer = SGDTrainer(model, loss_fn, train_loader, test_loader, epochs=SGD_EPOCHS, device=DEVICE, lr=LR)

    # from trainers.svrg import SVRGTrainer
    # trainer = SVRGTrainer(model, loss_fn, train_loader, test_loader, epochs=int(np.ceil(SGD_EPOCHS/(2+ FREQ_FG))), device=DEVICE, lr=LR)

    # from trainers.shuffle_svrg import ShuffleSVRGTrainer
    # trainer = ShuffleSVRGTrainer(model, loss_fn, train_loader, test_loader, epochs=int(np.ceil(SGD_EPOCHS/(2+ FREQ_FG))), device=DEVICE, lr=LR)


    from trainers.nfg_svrg import NFGSVRGTrainer
    FREQ_FG = 0
    SGD_EPOCHS = 15
    trainer = NFGSVRGTrainer(model, loss_fn, train_loader, test_loader, epochs=int(np.ceil(SGD_EPOCHS/(2+ FREQ_FG))), device=DEVICE, lr=LR)

    trainer.train()
    trainer.dump_histories()
