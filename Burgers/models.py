from typing import Callable, Dict, Tuple
from functools import partial

import jax.numpy as jnp
import ml_collections
import optax
import jax
from flax.training.train_state import TrainState
from flax import linen as nn
from jax import random, vmap, grad, jit


def _create_optimizer(
    config: ml_collections.ConfigDict,
) -> optax.GradientTransformation:
    lr = optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=config.transition_steps,
        decay_rate=config.decay_rate,
    )

    optimizer = optax.adam(
        learning_rate=lr,
        b1=config.b1,
        b2=config.b2,
        eps=config.eps,
        eps_root=config.eps_root,
    )
    return optimizer


def _create_train_state(
    config: ml_collections.ConfigDict,
) -> TrainState:
    arch = config.arch
    data = config.data

    model = FNO1d(
        arch.width,
        data.out_channels,
        arch.modes,
        arch.activation,
        arch.num_layers,
        arch.lift_init,
        arch.proj_init,
        arch.layer_init,
    )

    dummy_input = jnp.ones((data.spatial_dims, data.in_channels))
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
        return loss.mean()

    @partial(jit, static_argnums=(0,))
    def step(
        self,
        state: TrainState,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> TrainState:
        grads = grad(self.loss)(state.params, state, batch)
        state = state.apply_gradients(grads=grads)
        return state


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
                rng, shape, minval=-scale, maxval=scale
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
    layer_init: Callable

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        spectral_conv = SpectralConv1d(self.width, self.modes)(x)
        local_conv = nn.Conv(
            self.width,
            kernel_size=(1,),
            kernel_init=self.layer_init,
            name="local",
        )(x)
        return self.activation(spectral_conv + local_conv)


class FNO1d(nn.Module):
    width: int
    out_channels: int
    modes: int
    activation: Callable
    num_layers: int
    lift_init: Callable
    proj_init: Callable
    layer_init: Callable

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        x = nn.Conv(
            self.width,
            kernel_size=(1,),
            kernel_init=self.lift_init,
            name="lifting",
        )(x)
        for _ in range(self.num_layers):
            x = FNOBlock1d(
                self.width, self.modes, self.activation, self.layer_init
            )(x)
        x = nn.Conv(
            self.out_channels,
            kernel_size=(1,),
            kernel_init=self.proj_init,
            name="projection",
        )(x)
        return x
