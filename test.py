import jax
import jax.numpy as jnp
from jax import random, jit
import time

key = random.PRNGKey(0)
x = random.normal(key, (4000, 4000))

@jit
def compute(x):
    return jnp.dot(x, x.T)

start = time.time()
y = compute(x).block_until_ready()
print("Time:", time.time() - start)
print("Device:", y.device)