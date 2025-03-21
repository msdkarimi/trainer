from base_trainer import BaseTrainer
import torch


class Trainer(BaseTrainer):
    def __init__(self, model, dataloaders, optimizer, criterion, **kwargs):
        super().__init__(model, dataloaders, optimizer, criterion, **kwargs)

