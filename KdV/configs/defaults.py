import jax.numpy as jnp
import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.use_wandb = True

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "kdv"
    wandb.name = "current_model"
    wandb.tags = None
    wandb.group = None

    # Simulation settings
    config.data = data = ml_collections.ConfigDict()
    data.nt = 140
    data.nx = 256
    data.L = 128
    data.T = 140

    # FNO Architecture
    config.arch = arch = ml_collections.ConfigDict()
    arch.modes = 16
    arch.width = 64
    arch.seed = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 16
    training.time_history = 20
    training.time_future = 20
    training.epochs = 5 #30 * 2 * data.nt
    training.seed = 1

    # Optimizer
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "adamw"
    optim.learning_rate = 0.001
    optim.b1 = 0.9
    optim.b2 = 0.999
    optim.eps = 1e-8
    optim.eps_root = 0.0
    optim.weight_decay = 0.01
    optim.scale = 0.4
    optim.boundaries = jnp.array([10, 1200, 2400, 3600])
    return config
