import torch
from trainer.base_trainer import BaseTrainer
from modules.autoencoder import AutoencoderKL
from modules.diffusion import GaussianDiffusion
from configs.train_model_config import get_first_stage_config, get_diffusion_config
from tqdm import tqdm
from einops import rearrange, repeat
from util.utils import make_grid, noise_like

class Trainer(BaseTrainer):
    def __init__(self, model, dataloaders, optimizer, criterion, lr,
                 vae_key='input_image',
                 condition_key='input_caption',
                 conditional=False,
                 vae_scale_factor=0.18215,
                 diffusion_logger_every=50,
                 **kwargs):
        super().__init__(model, dataloaders, optimizer, criterion, lr, **kwargs)

        self.vae = AutoencoderKL(**get_first_stage_config()).to(self.device)
        self.diffusion = GaussianDiffusion(**get_diffusion_config()).to(self.device)
        self.conditioning_model=None

        self.vae_key = vae_key
        self.condition_key = condition_key
        self.vae_scale_factor = vae_scale_factor
        self.conditional = conditional
        self.diffusion_logger_every=diffusion_logger_every

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



    def log_image(self, batch,
                  n_row=2, sample=True,
                  ddim_steps=None, ddim_eta=1.,
                  plot_denoise_rows=False,
                  plot_progressive_rows=False,
                  plot_diffusion_rows=False,
                  plot_reconstruction_rows=False,
                  return_input=False,):
        use_ddim = ddim_steps is not None
        _log = dict()

        x, c = batch['input_image'], batch['input_caption']
        x = x.cuda() # todo update here
        z = self.encode_image(x)
        _batch_size = z.shape[0]

        if self.conditional:
            c = self.conditioning_model(c)
        else:
            c = None


        if return_input:
            _log.update({'input': x})
        if plot_reconstruction_rows:
            reconstruction = self.decode_latent(z)
            _log.update({'reconstruction': reconstruction})
        if plot_diffusion_rows:
            _diffused_images = list()
            z_start = z
            for t in range(self.diffusion.num_timesteps):
                if t % self.diffusion_logger_every == 0 or t == self.diffusion.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=_batch_size)
                    # t = t.to(self.device).long() #todo use to device
                    t = t.cuda().long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.diffusion.q_sample(x_start=z_start, t=t, noise=noise)
                    _diffused_images.append(self.decode_latent(z_noisy))


            diffusion_row = torch.stack(_diffused_images)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            # _log["diffused_images"] = diffusion_grid
            _log.update({'diffused_images': diffusion_grid})
        if sample:
            # todo add later ema model
            samples, z_denoise_row = self.sample_log(c, _batch_size) # z_denoise_row is intermediates from T to 0
            x_samples = self.decode_latent(samples)
            # _log["samples"] = x_samples
            _log.update({'samples': x_samples})
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                # _log["denoise_row"] = denoise_grid
                _log.update({'denoise_row': denoise_grid})

        if plot_progressive_rows:
            pass
            # todo to be implemented
            # # with self.ema_scope("Plotting Progressives"):
            # img, progressives = self.progressive_denoising(c, shape=(self.diffusion.channels, self.diffusion.image_size, self.diffusion.image_size), batch_size=_batch_size)
            # prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation Grid Handler")
            # # _log["progressive_row"] = prog_row
            # _log.update({'progressive_row': prog_row})

        return _log



    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        # for visualization purposes only
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_latent(zd))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim=None, ddim_steps=None, **kwargs):
        # todo ddim to be implemented
        # if ddim:
        #     ddim_sampler = DDIMSampler(self)
        #     shape = (self.channels, self.image_size, self.image_size)
        #     samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
        #                                                  shape, cond, verbose=False, **kwargs)
        #
        # else:
        samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

    @torch.no_grad()
    def sample(self, cond, batch_size, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, x0=None, shape=None):

        if shape is None:
            shape = (batch_size, self.model.in_channels, self.model.image_size, self.model.image_size)

        return self.p_sample_loop(shape, condition=cond, timesteps=timesteps, verbose=verbose, x0=x0, x_T=x_T, return_intermediates=return_intermediates)


    @torch.no_grad()
    def p_sample_loop(self, shape, condition=None, timesteps=None, verbose=True, x_T=None, return_intermediates=False, x0=None, start_T=None, mask=None):
        device = self.diffusion.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.diffusion.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling <t>', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:  # TODO what is mask for
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, ts, condition=condition, clip_denoised=self.diffusion.clip_denoised)
            if mask is not None:
                img_orig = self.diffusion.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % self.diffusion_logger_every == 0 or i == timesteps - 1:
                intermediates.append(img)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def p_sample(self, x, t, condition=None, clip_denoised=False, repeat_noise=False, return_x0=False, temperature=1., noise_dropout=0.):
        """
        Sample x_{t-1} from the model at the given timestep.
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_recon = self.p_mean_variance(x=x,
                                                                          c=condition,
                                                                          t=t,
                                                                          clip_denoised=clip_denoised
                                                                          )

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x_recon
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, c, t,
                        clip_denoised: bool):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        based on models prediction of noise(aka eps), and given t, tries to predict the x_0, then computes the posterior of mu and var
        """

        model_output = self.model(x, t, c)

        if self.diffusion.parameterization == 'eps':
            x_recon = self.diffusion.predict_start_from_noise(x, t=t, noise=model_output)
        else:
            raise NotImplementedError('only epsilon prediction is implemented')
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.diffusion.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon






    @torch.no_grad()
    def encode_image(self, x):
        z = self.vae.encode(x).sample().detach()
        return z * self.vae_scale_factor
    @torch.no_grad()
    def decode_latent(self, z):
        z = 1. / self.vae_scale_factor * z
        return self.vae.decode(z)
