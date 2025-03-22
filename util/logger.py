import os
import sys
import logging
from termcolor import colored
from datetime import datetime
from util.image_logger import ImageLogger
from torch.utils.tensorboard import SummaryWriter

class CustomLogger(object):
    def __init__(self, log_dir , log_image_kwargs, log_on, frequency, logger_name='', rescale=True, clamp=True):
        self.model_logger, _run_name = build_logger(log_dir, logger_name)
        self.image_logger = ImageLogger(log_dir, _run_name, log_image_kwargs, frequency=frequency, rescale=rescale, log_on=log_on, clamp=clamp)
        self.writer = SummaryWriter(f'{log_dir}/{_run_name}/run')

    def __call__(self, logger, log_dict=None, batch=None, step=None):
        if hasattr(logger, '__self__'):
            self.image_logger(logger, batch, step)
        else:
            self._log_dict(log_dict.update({'step': step}))

    def _log_dict(self, log_dict):
        if log_dict is not None:
            self.model_logger.info(
                '\t'.join(f'{k}:{log_dict[k]}' for k in log_dict)
            )







def build_logger(output_dir, name=''):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    # create console handlers for master process
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    # create file handlers
    os.makedirs(output_dir, exist_ok=True)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H %M %S")
    _run_name = f'log_{formatted_datetime}'
    _run_file_name = f'log_{formatted_datetime}.txt'
    os.makedirs(os.path.join(output_dir, _run_name), exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_dir, _run_name, _run_file_name), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger, _run_name