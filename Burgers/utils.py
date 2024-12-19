from functools import partial
from typing import Tuple

import jax.numpy as jnp
from torch.utils.data import Dataset
from jax import random, jit


class DataGenerator(Dataset):
    def __init__(
        self,
        inputs: jnp.ndarray,
        outputs: jnp.ndarray,
        batch_size: int,
        key: jnp.ndarray,
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.N = outputs.shape[0]
        self.batch_size = batch_size
        self.key = key

    @partial(jit, static_argnums=(0,))
    def __data_generation(
        self,
        key: jnp.ndarray,
        inputs: jnp.ndarray,
        outputs: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        inputs = inputs[idx, ...]
        outputs = outputs[idx, ...]
        return inputs, outputs

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(
            self.key, self.inputs, self.outputs
        )
        return inputs, outputs
