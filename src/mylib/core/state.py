"""BilevelState: state container for bilevel optimization."""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import jax

from mylib.core.types import PyTree

OptState = Any  # optax optimizer state or None


@dataclass
class BilevelState:
    """State container for bilevel optimization process.

    Registered as a JAX pytree so that instances can be passed through
    ``jax.jit``, ``jax.grad``, etc.

    Pytree layout
    -------------
    Leaves  : ``params``, ``hyperparams``, ``inner_opt_state``,
              ``outer_opt_state``, and the *values* of ``metadata``
              (in sorted-key order).
    Aux data: ``step`` (int) and the sorted *keys* of ``metadata``
              (tuple of strings).  Both are Python scalars / tuples and
              are therefore treated as static by JAX.

    Attributes:
        params: Model parameters (inner level).
        hyperparams: Hyperparameters to optimize (outer level).
        inner_opt_state: State of inner optimizer.
        outer_opt_state: State of outer optimizer.
        step: Current optimization step.
        metadata: Additional information (losses, norms, etc.).
                  Values may be JAX arrays or plain Python scalars.
    """
    params: PyTree
    hyperparams: PyTree
    inner_opt_state: OptState
    outer_opt_state: OptState
    step: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        params: PyTree,
        hyperparams: PyTree,
        inner_opt_state: OptState,
        outer_opt_state: OptState,
    ) -> "BilevelState":
        """Create a new BilevelState with initial values.

        Args:
            params: Initial model parameters.
            hyperparams: Initial hyperparameters.
            inner_opt_state: Initial inner optimizer state.
            outer_opt_state: Initial outer optimizer state.

        Returns:
            Initialized state object.
        """
        return cls(
            params=params,
            hyperparams=hyperparams,
            inner_opt_state=inner_opt_state,
            outer_opt_state=outer_opt_state,
            step=0,
            metadata={},
        )

    def update(
        self,
        params: Optional[PyTree] = None,
        hyperparams: Optional[PyTree] = None,
        inner_opt_state: Optional[OptState] = None,
        outer_opt_state: Optional[OptState] = None,
        step: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BilevelState":
        """Create updated state with new values.

        Args:
            params: New parameters (if None, keep current).
            hyperparams: New hyperparameters (if None, keep current).
            inner_opt_state: New inner optimizer state.
            outer_opt_state: New outer optimizer state.
            step: New step count.
            metadata: New metadata to merge.

        Returns:
            New state object with updates.
        """
        merged_metadata = dict(self.metadata)
        if metadata is not None:
            merged_metadata.update(metadata)

        return replace(
            self,
            params=params if params is not None else self.params,
            hyperparams=hyperparams if hyperparams is not None else self.hyperparams,
            inner_opt_state=(inner_opt_state if inner_opt_state is not None
                             else self.inner_opt_state),
            outer_opt_state=(outer_opt_state if outer_opt_state is not None
                             else self.outer_opt_state),
            step=step if step is not None else self.step,
            metadata=merged_metadata,
        )

    def get_metric(self, key: str, default: Any = None) -> Any:
        """Retrieve a metric from metadata.

        Args:
            key: Metric name.
            default: Default value if key not found.

        Returns:
            Metric value or default.
        """
        return self.metadata.get(key, default)


# ---------------------------------------------------------------------------
# JAX pytree registration
# ---------------------------------------------------------------------------
# Leaves  : params, hyperparams, inner_opt_state, outer_opt_state, and the
#           *values* of metadata (sorted by key) — all JAX-compatible.
# Aux data: step (int) and the sorted metadata *keys* (tuple of str) —
#           both are Python objects treated as static by JAX.
#
# This design allows metadata values to be JAX arrays (e.g. loss scalars
# returned inside jax.jit) while keeping the key names static.

def _bilevel_state_flatten(state: BilevelState):
    meta_keys = tuple(sorted(state.metadata.keys()))
    meta_vals = [state.metadata[k] for k in meta_keys]
    leaves = [
        state.params,
        state.hyperparams,
        state.inner_opt_state,
        state.outer_opt_state,
    ] + meta_vals
    aux = (state.step, meta_keys)
    return leaves, aux


def _bilevel_state_unflatten(aux, leaves):
    step, meta_keys = aux
    params, hyperparams, inner_opt_state, outer_opt_state = leaves[:4]
    meta_vals = leaves[4:]
    metadata = dict(zip(meta_keys, meta_vals))
    return BilevelState(
        params=params,
        hyperparams=hyperparams,
        inner_opt_state=inner_opt_state,
        outer_opt_state=outer_opt_state,
        step=step,
        metadata=metadata,
    )


jax.tree_util.register_pytree_node(
    BilevelState,
    _bilevel_state_flatten,
    _bilevel_state_unflatten,
)
