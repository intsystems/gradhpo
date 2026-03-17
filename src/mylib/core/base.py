"""BilevelOptimizer: abstract base class for bilevel hyperparameter optimizers."""

from abc import ABC, abstractmethod
from typing import Optional

import optax

from mylib.core.types import PyTree, LossFn, DataBatch
from mylib.core.state import BilevelState


class BilevelOptimizer(ABC):
    """Abstract base class for bilevel hyperparameter optimizers.

    All algorithms inherit from this class and implement
    the required abstract methods.

    Attributes:
        inner_optimizer: Optax optimizer for parameters.
        outer_optimizer: Optax optimizer for hyperparameters.
    """

    def __init__(
        self,
        inner_optimizer: Optional[optax.GradientTransformation] = None,
        outer_optimizer: Optional[optax.GradientTransformation] = None,
    ):
        """Initialize bilevel optimizer."""
        self.inner_optimizer = inner_optimizer
        self.outer_optimizer = outer_optimizer

    @abstractmethod
    def init(
        self,
        params: PyTree,
        hyperparams: PyTree,
    ) -> BilevelState:
        """Initialize the bilevel optimization state.

        Args:
            params: Initial model parameters.
            hyperparams: Initial hyperparameters.

        Returns:
            Initial optimization state.
        """

    @abstractmethod
    def step(
        self,
        state: BilevelState,
        train_batch: DataBatch,
        val_batch: DataBatch,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
    ) -> BilevelState:
        """Perform one bilevel optimization step.

        Args:
            state: Current bilevel state.
            train_batch: Training data batch.
            val_batch: Validation data batch.
            train_loss_fn: Training loss function.
            val_loss_fn: Validation loss function.

        Returns:
            Updated state.
        """

    @abstractmethod
    def compute_hypergradient(
        self,
        state: BilevelState,
        train_batch: DataBatch,
        val_batch: DataBatch,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
    ) -> PyTree:
        """Compute hypergradient w.r.t. hyperparameters.

        Args:
            state: Current state.
            train_batch: Training batch.
            val_batch: Validation batch.
            train_loss_fn: Training loss.
            val_loss_fn: Validation loss.

        Returns:
            Hypergradient with same structure as hyperparams.
        """
