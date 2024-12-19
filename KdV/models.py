from functools import partial
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import jit, random, value_and_grad, vmap


def _create_optimizer(
    config: ml_collections.ConfigDict,
) -> optax.GradientTransformation:

    learning_rate = config.learning_rate

    keys_arr = config.boundaries
    vals_arr = learning_rate * config.scale ** jnp.arange(
        1, keys_arr.shape[0] + 1
    )

    def dictionary_based_schedule(step, keys_arr, vals_arr, default_lr):
        idx = jnp.sum(keys_arr <= step) - 1
        return jnp.where(idx < 0, default_lr, vals_arr[idx])

    lr = lambda step: dictionary_based_schedule(
        step, keys_arr, vals_arr, learning_rate
    )

    optimizer = optax.adamw(
        learning_rate=lr,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
        weight_decay=config.weight_decay,
    )
    return optimizer


def _create_train_state(
    config: ml_collections.ConfigDict,
) -> TrainState:
    arch = config.arch
    training = config.training

    time_future = training.time_future
    time_history = training.time_history

    model = FNO1d(width=arch.width, modes=arch.modes, time_future=time_future)

    dummy_input = jnp.ones((config.data.nx, time_history))
    key = random.PRNGKey(arch.seed)
    params = model.init(key, dummy_input)

    tx = _create_optimizer(config.optim)

    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


class FNO:
    def __init__(
        self,
        config: ml_collections.ConfigDict,
    ):
        self.config = config
        self.state = _create_train_state(config)

    def loss(
        self,
        params: Dict,
        state: TrainState,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> jnp.ndarray:
        data, labels = batch
        pred = vmap(lambda x: state.apply_fn(params, x))(data)
        loss = jnp.square(pred - labels)
        return loss.sum()

    @partial(jit, static_argnums=(0,))
    def step(
        self,
        state: TrainState,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> TrainState:
        loss, grads = value_and_grad(self.loss)(state.params, state, batch)
        state = state.apply_gradients(grads=grads)
        return state, loss


class SpectralConv1d(nn.Module):
    width: int
    modes: int

    def setup(
        self,
    ):
        scale = 1 / (self.width * self.width)
        self.weights = self.param(
            "global_kernel",
            lambda rng, shape: random.uniform(
                rng, shape, minval=0, maxval=scale
            ),
            (2, self.modes, self.width, self.width),
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        spatial_resolution = x.shape[0]

        x_ft = jnp.fft.rfft(x, axis=0)
        x_ft_trunc = x_ft[: self.modes, :]

        R = jax.lax.complex(self.weights[0, ...], self.weights[1, ...])

        R_x_ft = jnp.einsum("Mio,Mi->Mo", R, x_ft_trunc)

        result = jnp.zeros((x_ft.shape[0], self.width), dtype=x_ft.dtype)
        result = result.at[: self.modes, :].set(R_x_ft)

        inv_ft_R_x_ft = jnp.fft.irfft(result, n=spatial_resolution, axis=0)
        return inv_ft_R_x_ft


class FNOBlock1d(nn.Module):
    width: int
    modes: int
    activation: Callable

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        spectral_conv = SpectralConv1d(self.width, self.modes)(x)
        local_conv = nn.Conv(self.width, kernel_size=(1,), name="local")(x)
        return self.activation(spectral_conv + local_conv)


class FNO1d(nn.Module):
    width: int
    modes: int
    time_future: int

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        x = nn.Dense(self.width)(x)

        x = FNOBlock1d(self.width, self.modes, nn.gelu)(x)
        x = FNOBlock1d(self.width, self.modes, nn.gelu)(x)
        x = FNOBlock1d(self.width, self.modes, nn.gelu)(x)
        x = FNOBlock1d(self.width, self.modes, nn.gelu)(x)

        x = nn.Dense(128)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.time_future)(x)
        return x
