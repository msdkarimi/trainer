from ml_collections import ConfigDict
# __all__ = ['get_all_config', 'get_config_trainer']
BATCH_SIZE = 4
NUM_ITER = 5e5
BASE_LR = 7e-5




def get_log_image_config():
    log_image_config = ConfigDict(
    {
        'n_row': 2, 'sample': True,
        'ddim_steps': None, 'ddim_eta': 1.,
        'plot_reconstruction_rows': True, 'plot_denoise_rows': False,
        'plot_progressive_rows': False, 'plot_diffusion_rows': False, 'return_input': False
    })
    return log_image_config

def get_image_logger_config():
    image_logger_config = ConfigDict()
    return image_logger_config




def get_trainer_config(input_config=None):
    if input_config is None:
        trainer_config = ConfigDict()
    else:
        trainer_config = ConfigDict(input_config)

    trainer_config.num_iter = NUM_ITER
    trainer_config.iter_type = 'step'
    trainer_config.log_every = 1000 # this interval is for validation,
                                    # other logs intervals like image logger and model logger interval
                                    # could be a multiplication of this parameter.
    trainer_config.metrics_to_monitor = ['train_loss', 'val_loss', 'grad_norm', 'param_norm', 'lr']
    trainer_config.checkpointing_dir = 'unet_checkpoints'
    trainer_config.diffusion_logger_every = 50

    return trainer_config






def get_optimizer_config():
    optimizer_config = ConfigDict()
    optimizer_config.name = 'Adam'
    optimizer_config.lr = BASE_LR
    optimizer_config.momentum = None
    optimizer_config.weight_decay = 0.0
    return optimizer_config

def get_latent_diffusion_config():
    latent_diffusion_configs = ConfigDict()
    latent_diffusion_configs.scale_factor = 0.18215
    latent_diffusion_configs.use_spatial_transformer = False
    latent_diffusion_configs.model_logger_every = 100
    latent_diffusion_configs.diffusion_logger_every = 50
    latent_diffusion_configs.lr_anneal_steps = NUM_ITER
    return latent_diffusion_configs




def get_unet_config():
    unet_configs = ConfigDict()
    unet_configs.image_size = 32
    unet_configs.in_channels = 4
    unet_configs.out_channels = 4
    unet_configs.model_channels = 320 #224
    unet_configs.attention_resolutions = [4, 2, 1] # [8, 4, 2]
    unet_configs.num_res_blocks = 2
    unet_configs.channel_mult = [1, 2, 4, 4]
    unet_configs.num_head_channels = 32
    # unet_configs.num_heads = 8 # instead of above
    # unet_configs.use_spatial_transformer = True
    # unet_configs.transformer_depth = 1
    # unet_configs.context_dim = 768
    return unet_configs

def get_diffusion_config():
    diffusion_configs = ConfigDict()
    diffusion_configs.linear_start = 0.00085 # 0.0015
    diffusion_configs.linear_end = 0.0120 # 0.0195
    diffusion_configs.timesteps = 1000
    diffusion_configs.beta_schedule = 'linear'
    diffusion_configs.loss_type = 'l1'
    diffusion_configs.first_stage_key = 'image'
    # diffusion_configs.cond_stage_key = 'caption'
    # diffusion_configs.image_size = 64
    diffusion_configs.image_size = 32
    # diffusion_configs.channels = 3
    diffusion_configs.channels = 4
    # diffusion_configs.conditioning_key = 'crossattn'
    diffusion_configs.monitor = 'val / loss_simple_ema'
    diffusion_configs.use_ema = False
    diffusion_configs.clip_denoised = True
    diffusion_configs.l_simple_weight = 1.
    diffusion_configs.use_positional_encodings = False
    diffusion_configs.learn_logvar = False
    diffusion_configs.logvar_init = 0.
    diffusion_configs.parameterization="eps"  # all assuming fixed variance schedules
    return diffusion_configs

def get_first_stage_config():
    first_stage_configs = ConfigDict()
    # first_stage_configs.name = 'autoencoder'
    first_stage_configs.embed_dim = 4
    first_stage_configs.ckpt_path = "pretrained/vae_f_8.ckpt"  # None
    # first_stage_configs.ckpt_path = "../pretrained/vae_f_8.ckpt"  # None
    # first_stage_configs.monitor: val / rec_loss
    first_stage_configs.ddconfig = ConfigDict()
    first_stage_configs.ddconfig.double_z = True
    first_stage_configs.ddconfig.z_channels = 4
    first_stage_configs.ddconfig.resolution = 256
    first_stage_configs.ddconfig.in_channels = 3
    first_stage_configs.ddconfig.out_ch = 3
    first_stage_configs.ddconfig.ch = 128
    first_stage_configs.ddconfig.ch_mult = [1, 2, 4, 4]
    first_stage_configs.ddconfig.num_res_blocks = 2
    first_stage_configs.ddconfig.attn_resolutions = []
    first_stage_configs.ddconfig.dropout = 0.0
    return first_stage_configs



if __name__ == '__main__':
    pass
    exit(0)