from base_trainer import BaseTrainer
from modules.autoencoder import AutoencoderKL
from modules.diffusion import GaussianDiffusion
from configs.train_model_config import get_first_stage_config, get_diffusion_config

class Trainer(BaseTrainer):
    def __init__(self, model, dataloaders, optimizer, criterion, lr, **kwargs):
        super().__init__(model, dataloaders, optimizer, criterion, lr, **kwargs)

        self.vae = AutoencoderKL(**get_first_stage_config())
        self.diffusion = GaussianDiffusion(**get_diffusion_config())


    def __call__(self, *args, **kwargs):
        pass





