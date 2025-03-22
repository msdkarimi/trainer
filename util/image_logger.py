import os
import torch
import torchvision
import numpy as np
from PIL import Image

class ImageLogger(object):
    def __init__(self, logger_name, logger_folder, log_image_kwargs, frequency=500, rescale=True, log_on='step', clamp=True):
        self.frequency = frequency
        self.save_dir = [logger_name, logger_folder,]
        self.rescale = rescale
        self.log_on = log_on
        self.clamp = clamp
        self.log_image_kwargs = log_image_kwargs

    def __call__(self, model, batch, step):
        def _log_images():
            with torch.no_grad():
                images = model.log_image(batch, **self.log_image_kwargs)

            for key in images:  # key could be dict_keys(['inputs', 'reconstructions', 'conditionings', 'diffused_images', 'samples', 'progressive_row'])
                if isinstance(images[key], torch.Tensor):
                    images[key] = images[key].detach().cpu()
                    if self.clamp:
                        images[key] = torch.clamp(images[key], -1., 1.)

            for k in images:
                grid = torchvision.utils.make_grid(images[k], nrow=2)
                if self.rescale:
                    grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid * 255).astype(np.uint8)
                # os.makedirs(os.path.split(path), exist_ok=True)
                os.makedirs(os.path.join(*self.save_dir, 'images', k,), exist_ok=True)
                filename = f'{step:012}.png'
                path = os.path.join(*self.save_dir, 'images', k, filename)
                Image.fromarray(grid).save(path)

        if self.log_on == 'step' and  step % self.frequency == 0:
            _train_phase = model.model.training
            if _train_phase:
                model.model.eval()
            _log_images()
            if _train_phase:
                model.model.train()
