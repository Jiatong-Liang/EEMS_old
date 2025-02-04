# This utility_methods file is just a collection of helper functions

import numpy as np
import jax.numpy as jnp
import random

# function that chunks the het matrix
def _my_chunk_het_matrix(
    het_matrix: np.ndarray,
    overlap: int,
    chunk_size: int,
) -> np.ndarray:
    data = het_matrix.clip(-1, 1).astype(np.int8)
    assert data.ndim == 2
    data = np.ascontiguousarray(data)
    assert data.data.c_contiguous
    N, L = data.shape
    S = chunk_size + overlap
    L_pad = int(np.ceil(L / S) * S)
    padded_data = np.pad(data, [[0, 0], [0, L_pad - L]], constant_values=-1)
    assert L_pad % S == 0
    num_chunks = L_pad // S
    new_shape = (N, num_chunks, S)
    new_strides = (
        padded_data.strides[0],
        padded_data.strides[1] * chunk_size,
        padded_data.strides[1],
    )
    chunked = np.lib.stride_tricks.as_strided(
        padded_data, shape=new_shape, strides=new_strides
    )
    return np.copy(chunked.reshape(-1, S)), new_shape

# function that calls on _my_chunk_het_matrix.
# Tou need to get the het matrix first, then the chunk size, and then
# get the final result of the chunked het matrix
def chunk_attributes(data, window_size, overlap):
    tmp1 = data.get_data(window_size)['het_matrix']
    print(tmp1)
    chunk_size1 = int(data.L / (5 * window_size))
    print(chunk_size1)
    tmp2, shape = _my_chunk_het_matrix(tmp1, overlap, chunk_size1)
    print(shape)
    return shape

# Creates dictionary mappings, the pairing is individual : chunk index
# and individual : vertex, where individual is a tuple pair corresponding
# to the ARG nodes associated to the individual
# This returns a mapping from vertex: an array of arrays of chunk indices
# def chunk_mapping(data, ts, num_chunks):
#     dictionary1 = {}
#     dictionary2 = {}
#     dictionary3 = {}
#     nodes = data.nodes
#     for i in range(len(nodes)):
#         dictionary1[nodes[i]] = jnp.arange((i * num_chunks), (i * num_chunks) + num_chunks)
#         # both arg ids will correspond to the same individual sampled
#         individual1_arg1, individual1_arg2 = nodes[i]
#         dictionary2[nodes[i]] = (ts.node(individual1_arg1).population, ts.node(individual1_arg2).population)
    
#     # Iterate over the second dictionary and use its values as keys in the new dict
#     for key, new_key in dictionary2.items():
#         if key in dictionary1:
#             if new_key not in dictionary3:
#                 dictionary3[new_key] = []
#             dictionary3[new_key].append(dictionary1[key])

#     return dictionary1, dictionary2, dictionary3


# This method maps chunk indices to the vertices
def chunk_mapping(nodes, ts, num_chunks):
    dictionary1 = {}
    dictionary2 = {}
    dictionary3 = {}
    for i in range(len(nodes)):
        dictionary1[nodes[i]] = jnp.arange((i * num_chunks), (i * num_chunks) + num_chunks)
        # both arg ids will correspond to the same individual sampled
        individual1_arg1, individual1_arg2 = nodes[i]
        dictionary2[nodes[i]] = (ts.node(individual1_arg1).population, ts.node(individual1_arg2).population)

    # Step 2: Iterate through the first dictionary
    for key1, value1 in dictionary1.items():
        # Convert the size 5 NumPy array to a tuple
        # value1_tuple = tuple(value1.tolist())
        
        # Step 3: For each value in dict1, find all matching tuples in dict2
        for key2, value2 in dictionary2.items():
            if key1 == key2:
                # Use the tuple from dict1 as the key and the tuple from dict2 as the value
                dictionary3[value2] = value1

    return dictionary1, dictionary2, dictionary3

def chunk_map(nodes, num_chunks):
    dictionary = {}
    for i in range(len(nodes)):
        dictionary[nodes[i]] = jnp.arange((i * num_chunks), (i * num_chunks) + num_chunks)
    return dictionary

# This function will pull two random individuals aka tuple pairs from a 
# dictionary
def random_individuals(arr, num = 2):
    """
    Picks two random keys from a dictionary.

    Parameters
    ----------
    d : dict
        The input dictionary.

    Returns
    -------
    list
        A list containing two randomly selected keys from the dictionary.
    """
    if len(arr) < 2:
        raise ValueError("The dictionary must have at least two keys to pick from.")
    
    return random.sample(arr, 2)

def random_chromosome_pair(arr, num):
    """
    Picks two random keys from a dictionary.

    Parameters
    ----------
    d : dict
        The input dictionary.

    Returns
    -------
    list
        A list containing two randomly selected keys from the dictionary.
    """
    if len(arr) < 1:
        raise ValueError("The dictionary must have at least two keys to pick from.")
    
    return random.sample(arr, num)

import jax
# def pull_random_pairs(dictionary, n, subkey = jax.random.PRNGKey(0)):
#     # Get the keys and values
#     keys = list(dictionary.keys())
    
#     # Generate random indices with replacement
#     key_indices = jax.random.randint(subkey, shape=(n,), minval=0, maxval=len(keys))
    
#     # Pull out random keys and values using the random indices
#     random_keys = [keys[i] for i in key_indices]
#     # Get associated arrays for the selected keys
#     random_arrays = []
#     for key in random_keys:
#         # Randomly select an index for the arrays in the list associated with this key
#         array_index = jax.random.randint(subkey, shape=(), minval=0, maxval=len(dictionary[key]))
#         random_arrays.append(dictionary[key][array_index])
        
#     return random_keys, random_arrays

def pull_random_pairs(dictionary, n, subkey = jax.random.PRNGKey(0)):
    # Get the keys and values
    keys = jnp.array(list(dictionary.keys()))
    values = jnp.array(list(dictionary.values()))
    
    # Generate random indices with replacement
    key_indices = jax.random.randint(subkey, shape=(n,), minval=0, maxval=len(keys))
    
    # Pull out random keys and values using the random indices
    random_keys = keys[key_indices]
    random_values = values[key_indices]

    # values need to be returned first because you pass init_vertices first, then the random keys are the chunk indices
    return jnp.array(random_values), jnp.array(random_keys)

### This function can be deleted later, it exists only to plot things to test out
### values of q
import matplotlib.pyplot as plt

def plotting(y1, y2):
    """
    Plots two sets of y-coordinates on the same graph with x-coordinates defaulting to integers.

    Parameters
    ----------
    y1 : list of int or float
        The first set of y-coordinates.
    y2 : list of int or float
        The second set of y-coordinates.
    """
    x1 = range(len(y1))  # Default x-coordinates for the first set of y-coordinates
    x2 = range(len(y2))  # Default x-coordinates for the second set of y-coordinates

    plt.figure(figsize=(8, 6))  # Create a new figure with a specified size

    plt.plot(x1, y1, 'bo-', label='True q')  # Plot the first set of y-coordinates with blue circles and lines
    plt.plot(x2, y2, 'rs-', label='MCMC q')  # Plot the second set of y-coordinates with red squares and lines

    plt.title('Plot of Multiple Y-coordinates')
    plt.xlabel('X-axis (default integers)')
    plt.ylabel('Y-axis')

    plt.legend()  # Show legend
    plt.grid(True)  # Show grid
    plt.show()  # Display the plot