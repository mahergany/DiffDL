from inception import InceptionV3
import os
import numpy as np
import jax
import jax.numpy as jnp
import functools

def get_activations():
    key = jax.random.PRNGKey(0)

    model = InceptionV3(pretrained=True)
    params = model.init(key, jnp.ones((1,256,256,3)))

    apply_fn = jax.jit()

def get_data_distribution():
    pass