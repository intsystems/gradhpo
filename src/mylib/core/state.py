"""BilevelState: state container for bilevel optimization."""

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

from mylib.core.types import PyTree

OptState = Any  # optax optimizer state or None


@dataclass
class BilevelState:
    """State container for bilevel optimization process.

    Attributes:
        params: Model parameters (inner level).
        hyperparams: Hyperparameters to optimize (outer level).
        inner_opt_state: State of inner optimizer.
        outer_opt_state: State of outer optimizer.
        step: Current optimization step.
        metadata: Additional information (losses, norms, etc.).
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
