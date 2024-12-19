from typing import List, Tuple

import jax.numpy as jnp
import ml_collections
import numpy as npy
from flax.training.train_state import TrainState
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run
from tqdm import trange
from jax import random

from utils import create_data, create_dataloader
from models import FNO


def train_model(
    config: ml_collections.ConfigDict,
    run: Run = None,
) -> TrainState:
    def train(
        state: TrainState,
        key: jnp.ndarray,
        loader: DataLoader,
    ) -> Tuple[TrainState, List, jnp.ndarray]:
        time_history = training.time_history
        time_future = training.time_future
        batch_size = training.batch_size

        max_start_time = (config.data.nt - time_history) - time_future
        possible_start_times = jnp.arange(
            time_history, max_start_time + 1, time_history
        )

        losses = npy.zeros(len(loader))
        for i, (u, _, _) in enumerate(loader):
            key, subkey = random.split(key)
            start_time = random.choice(
                subkey, possible_start_times, (batch_size,)
            )

            data, labels = create_data(
                u, start_time, time_history, time_future
            )

            batch = (
                jnp.permute_dims(data, (0, 2, 1)),
                jnp.permute_dims(labels, (0, 2, 1)),
            )
            state, loss = model.step(state, batch)
            losses[i] = loss.item()
        return state, losses / batch_size, key

    training = config.training

    train_loader = create_dataloader(
        "./data/KdV_train.h5",
        mode="train",
        nt=config.data.nt,
        nx=config.data.nx,
        batch_size=training.batch_size,
    )

    epochs = training.epochs
    pbar = trange(epochs)

    model = FNO(config)

    key = random.PRNGKey(training.seed)
    for epoch in pbar:
        model.state, losses, key = train(model.state, key, train_loader)

        if (epoch % 10 == 0) or (epoch == epochs):
            loss = losses.mean()
            pbar.set_postfix({"train_loss": loss})
            if run is not None:
                run.log({"train_loss": loss})
    return model.state
