from abc import ABC, abstractmethod
from typing import Tuple, Literal
from util.utils import AverageMeter
import torch
from torch import nn
from torch.utils.data import DataLoader
import os


class BaseTrainer(ABC):
    @abstractmethod
    def __init__(self, model:nn.Module,
                 dataloaders:Tuple[DataLoader, DataLoader],
                 optimizer,
                 criterion,
                 num_iter:int=5000,
                 iter_type: Literal['epoch', 'step']='epoch',
                 lr_scheduler=None,
                 device='cuda',
                 logger=None,
                 log_every=2,
                 metrics_to_monitor=None,
                 checkpointing_dir=None
                 ) -> None:

        self.device = device
        self.model = self.push_to_device(model)
        self.train_dataloader, self.val_dataloader = dataloaders
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_iter = num_iter
        self.num_iter_init = 0
        self.iter_type = iter_type
        self.lr_scheduler = lr_scheduler
        self.logger = logger
        self.log_every = log_every
        self.metrics_to_monitor = {'loss': AverageMeter()} if metrics_to_monitor is None else self._update_metrics_to_monitor(metrics_to_monitor)
        self.checkpointing_dir=checkpointing_dir or 'checkpoints_dir'
        os.makedirs(self.checkpointing_dir, exist_ok=True)
        self.load_checkpoint()
    @staticmethod
    def _update_metrics_to_monitor(metrics):
        return { metric: AverageMeter for metric in metrics}

    def take_one_step(self,args, kwargs):
        """
        :param args: contains all the inputs that model requires
        :param kwargs: contains other essential information that need to complete learning, e.g, ground truth,...
        :return:
        """

        model_output = self.forward_pass(args)
        losses = self.compute_loss(model_output, kwargs)
        self._update_meters(losses)
        if self.model.training:
            self.backpropagation(losses)
            if self.lr_scheduler is not None: self.lr_scheduler.step()
        if self.logger is not None: self.logger(losses)

    def _update_meters(self, losses):
        for _loss in self.metrics_to_monitor:
            self.metrics_to_monitor[_loss].update(losses[_loss].item())

    def forward_pass(self, args, **kwargs):
        return self.model(*args, **kwargs)

    def backpropagation(self, losses):
        _loss = losses['loss']
        self.optimizer.zero_grad()
        _loss.backward()
        self.optimizer.step()


    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor):
        return self.criterion(predictions, targets)

    def validate(self,):
        pass
    def log_model(self,):
        pass

    def __call__(self,**kwargs):
        if self.iter_type=='epoch':
            self._epoch_wise(**kwargs)
        elif self.iter_type=='step':
            self._step_wise(**kwargs)

    def push_to_device(self, x):
        assert self.device is not None, 'the device must be specified either cpu or cuda!'
        return x.to(self.device)

    def separate_input_from_gt(self, x):
        """
        dataloader must output dict in a way that its keys are arranged in a way that the usage of data is specified
         by the name of its key
        :param x: a dict
        :return: input, gts
        """
        _input = set()
        _gt = dict()

        for key in x:
            if 'input' in key:
                _i = x[key]
                if type(_i) == torch.Tensor:
                    _i = self.push_to_device(_i)
                _input.add(_i)
            else:
                _i = x[key]
                if type(_i) == torch.Tensor:
                    _i = self.push_to_device(_i)
                _gt.update({key: _i})
        return _input, _gt

    def _epoch_wise(self, **kwargs):
        def _one_epoch(**kwargs):
            _data = self.train_dataloader if self.model.training else self.val_dataloader
            for batch_idx, batch in enumerate(_data):
                batch = {'input': batch[0], 'gt': batch[1]}
                inputs, gts = self.separate_input_from_gt(batch)
                gts.update(**kwargs)
                self.take_one_step(inputs, gts)
        for epoch in range(self.num_iter_init, self.num_iter):

            _one_epoch(**kwargs)
            if epoch % self.log_every == 0:
                self.num_iter_init = epoch
                print(self.metrics_to_monitor['loss'].avg)
                self.checkpointing()
                self.validate()

    def _step_wise(self, **kwargs):
        _data = self.train_dataloader if self.model.training else self.val_dataloader
        _reinit = _data
        while self.num_iter_init <= self.num_iter:
            try:
                batch = next(iter(_data))
                batch = {'input': batch[0], 'gt': batch[1]}
            except StopIteration:
                _data = _reinit
                batch = next(iter(_data))
                batch = {'input': batch[0], 'gt': batch[1]}

            inputs, gts = self.separate_input_from_gt(batch)
            gts.update(**kwargs)
            self.take_one_step(inputs, gts)
            self.num_iter_init += 1

    def checkpointing(self, just_weights=False, name=''):
        _dir = self.checkpointing_dir
        _name = name or 'model'
        if just_weights:
            torch.save({'state_dict':self.model.state_dict()}, os.path.join(_dir, f'{_name}.pt'))
        else:
            _checkpoint = {
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'num_iter_init': self.num_iter_init,
                'lr': self.optimizer.param_groups[0]['lr'],
                'loss': self.metrics_to_monitor['loss'].avg,
            }
            torch.save(_checkpoint, os.path.join(_dir, f'{_name}.pt'))

        if self.logger is not None:
            self.logger.info(f'save model {self.iter_type}={self.num_iter_init} @ {_dir}/{_name}.pt')
        else:
            print(f'save model {self.iter_type}={self.num_iter_init} @ {_dir}/{_name}.pt')

    def load_checkpoint(self, just_weights=False, name='model.pt'):
        try:
           _ckpt = torch.load(os.path.join(self.checkpointing_dir, f'{name}'), map_location=self.device)
           self.model.load_state_dict(_ckpt['state_dict'])
           if not just_weights:
               self.optimizer.load_state_dict(_ckpt['optimizer'])
               self.metrics_to_monitor['loss'].update(_ckpt['loss'])
               self.num_iter_init = _ckpt['num_iter_init']

               if len(self.optimizer.param_groups) > 0:
                   self.optimizer.param_groups[0]['lr'] = _ckpt['lr']
           if self.logger is not None:
               self.logger.info(f'load model from {self.checkpointing_dir}/{name}, restoring from {self.iter_type}={self.num_iter_init}')
           else:
               print(f'load model from {self.checkpointing_dir}/{name}, restoring from {self.iter_type}={self.num_iter_init}')

        except FileNotFoundError:
            if self.logger is not None:
                self.logger.info(f'no checkpoint found @ {self.checkpointing_dir}/{name}, train from scratch!')
            else:
                print(f'no checkpoint found @ {self.checkpointing_dir}/{name}, train from scratch!')
        except KeyError as e:
            print(f"Warning: Missing key in checkpoint. Exception: {e}")
        except Exception as e:
            print(f"Warning: {e}")
