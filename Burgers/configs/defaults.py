import ml_collections
from flax import linen as nn


def get_config():
    config = ml_collections.ConfigDict()

    config.use_wandb = True

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "burgers"
    wandb.name = "sweep"
    wandb.tags = None
    wandb.group = None

    # Simulation Settings
    config.data = data = ml_collections.ConfigDict()
    data.in_channels = 2
    data.out_channels = 1
    data.spatial_dims = 2048

    # FNO Architecture
    config.arch = arch = ml_collections.ConfigDict()
    arch.modes = 24
    arch.width = 64
    arch.num_layers = 12
    arch.seed = 0
    arch.activation = nn.tanh
    arch.layer_init = nn.initializers.glorot_uniform()
    arch.lift_init = nn.initializers.glorot_uniform()
    arch.proj_init = nn.initializers.glorot_uniform()

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 128
    training.epochs = 4
    training.seed = 1

    # Optimizer
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "adam"
    optim.learning_rate = 1e-3
    optim.b1 = 0.9
    optim.b2 = 0.999
    optim.eps = 1e-8
    optim.eps_root = 0.0

    optim.transition_steps = 250
    optim.transition_begin = 0
    optim.decay_rate = 0.9
    return config
