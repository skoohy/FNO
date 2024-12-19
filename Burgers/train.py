import ml_collections
import numpy as npy
from flax.training.train_state import TrainState
from wandb.sdk.wandb_run import Run
from tqdm import trange
from jax import random

from utils import DataGenerator
from models import FNO

data = npy.load("./data/burgers_data.npy", allow_pickle=True).item()

a_with_mesh = data["a_with_mesh"]
u = data["u"]

train_split = int(0.75 * a_with_mesh.shape[0])

train_x = a_with_mesh[:train_split, ::4, :]
train_x_mean = train_x.mean((0, 1))
train_x_std = train_x.std((0, 1))
train_x = (train_x - train_x_mean) / train_x_std

train_y = u[:train_split, ::4, :]
train_y_mean = train_y.mean((0, 1))
train_y_std = train_y.std((0, 1))
train_y = (train_y - train_y_mean) / train_y_std


def train_model(
    config: ml_collections.ConfigDict,
    run: Run = None,
) -> TrainState:
    dataset = DataGenerator(
        train_x,
        train_y,
        config.training.batch_size,
        random.PRNGKey(config.training.seed),
    )

    epochs = config.training.epochs
    pbar = trange(epochs)
    data = iter(dataset)

    model = FNO(config)

    for epoch in pbar:
        batch = next(data)
        model.state = model.step(model.state, batch)

        if (epoch % 20 == 0) or (epoch == epochs):
            loss = model.loss(model.state.params, model.state, batch)
            pbar.set_postfix({"train_loss": loss})

            if run is not None:
                run.log({"train_loss": loss})

    return model.state
