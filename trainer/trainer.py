import torch
from trainer.base_trainer import BaseTrainer
from modules.autoencoder import AutoencoderKL
from modules.diffusion import GaussianDiffusion
from configs.train_model_config import get_first_stage_config, get_diffusion_config

class Trainer(BaseTrainer):
    def __init__(self, model, dataloaders, optimizer, criterion, lr,
                 vae_key='input_image',
                 condition_key='input_caption',
                 conditional=False,
                 vae_scale_factor=0.18215,
                 **kwargs):
        super().__init__(model, dataloaders, optimizer, criterion, lr, **kwargs)

        self.vae = AutoencoderKL(**get_first_stage_config()).to(self.device)
        self.diffusion = GaussianDiffusion(**get_diffusion_config()).to(self.device)
        self.conditioning_model=None

        self.vae_key = vae_key
        self.condition_key = condition_key
        self.vae_scale_factor = vae_scale_factor
        self.conditional = conditional

    def _step_wise(self, **kwargs):
        _data = self.train_dataloader if self.model.training else self.val_dataloader
        _reinit = _data
        while self.num_iter_init <= self.num_iter:
            _gt = dict()
            try:
                batch = next(iter(_data))
            except StopIteration:
                _data = _reinit
                batch = next(iter(_data))
            # inputs, gts = self.separate_input_from_gt(batch)
            z = self.encode_image(batch[self.vae_key].to(self.device))
            noise = torch.randn_like(z).to(self.device) #   the target
            t = torch.randint(0, self.diffusion.num_timesteps, (z.shape[0],), device=self.device).long()
            z_noisy = self.diffusion.q_sample(x_start=z, t=t, noise=noise)
            if self.conditional:
                c = self.conditioning_model(batch[self.condition_key])
            else:
                c = None


            _gt.update(**kwargs)
            _gt.update({'target_noise': noise})

            self.take_one_step((z_noisy, t, c), _gt)
            self.num_iter_init += 1
            self._log(batch)

    @torch.no_grad()
    def encode_image(self, x):
        z = self.vae.encode(x).sample().detach()
        return z * self.vae_scale_factor
    @torch.no_grad()
    def decode_image(self, z):
        z = 1. / self.vae_scale_factor * z
        return self.vae.decode(z)
