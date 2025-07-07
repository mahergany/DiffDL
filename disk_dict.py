import numpy as np
from time import time
import os
import jax.numpy as jnp
import jax

# def disk_dict(r_min,r_max):
#     """
#     function that creates a npy dictionnary of binary disk masks to speed up image generation.
#     r_min : minimal radius
#     r_max : maximal radius
#     """
#     disk_d = dict()
#     for r in range(r_min,r_max+1,1):
#         print(r)
#         t0 = time()
#         L = jnp.arange(-r,r + 1,dtype = np.int32)
#         X, Y = jnp.meshgrid(L, L)
#         disk_1d = jnp.array((X ** 2 + Y ** 2) <= r ** 2,dtype = bool)
#         disk_d[str(r)] = disk_1d
#         print(time()-t0)
#     if not(os.path.isdir("npy")):
#         os.makedirs("npy")
#     jnp.save("npy/dict_jnp.npy", disk_d)

# disk_dict(1,1000)


def disk_dict_array(r_min, r_max):
    """
    Create a 3D JAX array with all disk shapes stacked along axis=0
    Index i corresponds to radius = r_min + i
    """
    shapes = []
    for r in range(r_min, r_max + 1):
        print(f"Generating radius {r}")
        L = jnp.arange(-r, r + 1, dtype=jnp.int32)
        X, Y = jnp.meshgrid(L, L)
        disk = (X ** 2 + Y ** 2) <= r ** 2
        padded = jnp.zeros((2 * r_max + 1, 2 * r_max + 1), dtype=bool)
        # Center the disk in the padded array
        offset = r_max - r
        padded = padded.at[offset:offset + 2 * r + 1, offset:offset + 2 * r + 1].set(disk)
        shapes.append(padded)
    
    all_disks = jnp.stack(shapes, axis=0)  # Shape: (r_max - r_min + 1, 2*r_max+1, 2*r_max+1)
    jnp.save("npy/disk_array_jnp.npy", all_disks)

disk_dict_array(1,1000)
