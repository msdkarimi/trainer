import torch
from timm import optim

from data.data_loader import DareDataset
from torch.utils.data import DataLoader
from util.utils import image_transform

def get_data_loaders(root, batch_size):
    train_dataset = DareDataset(root, 'train', transform=image_transform)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    val_dataset = DareDataset(root, 'validation', transform=image_transform)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataset = DareDataset(root, 'test', transform=image_transform)

    return train_data_loader, val_data_loader

def get_optimizer(model, name=None, lr=None, momentum=None, weight_decay=None):
    if name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        return optimizer
    elif name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer
    else:
        raise NotImplementedError('currently supports just Adam and SGD')

def get_criterion(prediction, target, loss_type='l2'):
    if loss_type == 'l2':
        loss = torch.nn.functional.mse_loss(prediction, target)
    elif loss_type == 'l1':
        loss = (target - prediction).abs()
    else:
        raise NotImplementedError

    return loss
