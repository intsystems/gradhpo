"""Core type definitions for gradhpo.

Provides type aliases and data structures used across all algorithms.
"""

from typing import Any, Callable, Dict, NamedTuple, Union

import jax.numpy as jnp
from jax import Array

# Type aliases
PyTree = Any  # JAX PyTree (nested structure of arrays)
LossFn = Callable[[PyTree, PyTree, Any], Array]
MetricDict = Dict[str, Union[float, Array]]


class DataBatch(NamedTuple):
    """Structure for a data batch.

    Attributes:
        inputs: Input features [batch_size, ...]
        targets: Target labels [batch_size, ...]
    """
    inputs: Array
    targets: Array


class LossFunctions(NamedTuple):
    """Container for loss functions.

    Attributes:
        train_loss: Training loss function
        val_loss: Validation loss function
    """
    train_loss: LossFn
    val_loss: LossFn
