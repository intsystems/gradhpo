# gradhpo/algorithms/greedy.py

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax

from gradhpo.core.base import BilevelOptimizer
from gradhpo.core.state import BilevelState
from gradhpo.core.types import DataBatch, LossFn, PyTree


class GreedyOptimizer(BilevelOptimizer):
    """Generalized greedy gradient-based hyperparameter optimization.

    This implements the approximation from Eq. (6) of the attached paper:

        d_hat_alpha =
            ∇_alpha L_val(w_T, alpha)
            + sum_{t=1..T} gamma^(T-t) * ∇_{w_t} L_val(w_t, alpha) * B_t

    where B_t = ∂Φ(w_{t-1}, alpha) / ∂alpha and Φ is one inner optimizer step.

    Notes
    -----
    - `unroll_steps` corresponds to the horizon T.
    - `gamma` controls how strongly earlier greedy terms contribute.
    - A single `train_batch` / `val_batch` is reused across the unrolled inner steps,
      which matches the current library template API.
    """

    def __init__(
        self,
        inner_optimizer: optax.GradientTransformation,
        outer_optimizer: optax.GradientTransformation,
        unroll_steps: int = 1,
        gamma: float = 0.9,
    ):
        super().__init__(inner_optimizer, outer_optimizer)

        if unroll_steps < 1:
            raise ValueError(f"`unroll_steps` must be >= 1, got {unroll_steps}.")
        if not (0.0 < gamma <= 1.0):
            raise ValueError(f"`gamma` must be in (0, 1], got {gamma}.")

        self.unroll_steps = unroll_steps
        self.gamma = gamma

    def init(
        self,
        params: PyTree,
        hyperparams: PyTree,
    ) -> BilevelState:
        """Initialize Greedy optimization state."""
        inner_opt_state = self.inner_optimizer.init(params)
        outer_opt_state = self.outer_optimizer.init(hyperparams)

        return BilevelState.create(
            params=params,
            hyperparams=hyperparams,
            inner_opt_state=inner_opt_state,
            outer_opt_state=outer_opt_state,
        )

    def _one_inner_step(
        self,
        params: PyTree,
        hyperparams: PyTree,
        inner_opt_state: optax.OptState,
        train_batch: DataBatch,
        train_loss_fn: LossFn,
    ) -> Tuple[PyTree, optax.OptState, jax.Array]:
        """Perform one inner optimization step on train loss."""
        loss, grads = jax.value_and_grad(train_loss_fn)(
            params,
            hyperparams,
            train_batch,
        )
        updates, new_inner_opt_state = self.inner_optimizer.update(
            grads,
            inner_opt_state,
            params,
        )
        new_params = optax.apply_updates(params, updates)
        return new_params, new_inner_opt_state, loss

    def _rollout(
        self,
        state: BilevelState,
        train_batch: DataBatch,
        train_loss_fn: LossFn,
    ) -> Tuple[
        PyTree,
        optax.OptState,
        jax.Array,
        tuple[tuple[PyTree, optax.OptState, PyTree], ...],
    ]:
        """Unroll inner optimization for `self.unroll_steps`.

        Returns
        -------
        final_params:
            Parameters after the unrolled inner optimization.
        final_inner_opt_state:
            Inner optimizer state after rollout.
        last_train_loss:
            Training loss from the last inner step.
        transitions:
            Tuple of (pre_step_params, pre_step_inner_opt_state, post_step_params)
            for each inner step.
        """
        params = state.params
        inner_opt_state = state.inner_opt_state
        transitions = []
        last_train_loss = jnp.asarray(0.0)

        for _ in range(self.unroll_steps):
            pre_params = params
            pre_inner_opt_state = inner_opt_state

            params, inner_opt_state, last_train_loss = self._one_inner_step(
                params=params,
                hyperparams=state.hyperparams,
                inner_opt_state=inner_opt_state,
                train_batch=train_batch,
                train_loss_fn=train_loss_fn,
            )

            transitions.append((pre_params, pre_inner_opt_state, params))

        return params, inner_opt_state, last_train_loss, tuple(transitions)

    def _compute_hypergradient_from_rollout(
        self,
        hyperparams: PyTree,
        final_params: PyTree,
        transitions: tuple[tuple[PyTree, optax.OptState, PyTree], ...],
        train_batch: DataBatch,
        val_batch: DataBatch,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
    ) -> PyTree:
        """Compute Eq. (6) from the paper on top of a precomputed rollout."""
        # Direct outer gradient term: ∇_alpha L_val(w_T, alpha)
        hypergrad = jax.grad(
            lambda hp: val_loss_fn(final_params, hp, val_batch)
        )(hyperparams)

        total_steps = len(transitions)

        for t, (pre_params, pre_inner_opt_state, post_params) in enumerate(
            transitions, start=1
        ):
            weight = self.gamma ** (total_steps - t)

            # ∇_{w_t} L_val(w_t, alpha)
            val_grad_wt = jax.grad(
                lambda p: val_loss_fn(p, hyperparams, val_batch)
            )(post_params)

            weighted_val_grad_wt = jtu.tree_map(
                lambda g: weight * g,
                val_grad_wt,
            )

            # Local update map Φ(w_{t-1}, alpha) with w_{t-1} fixed
            def phi(hp: PyTree) -> PyTree:
                train_grads = jax.grad(train_loss_fn)(
                    pre_params,
                    hp,
                    train_batch,
                )
                updates, _ = self.inner_optimizer.update(
                    train_grads,
                    pre_inner_opt_state,
                    pre_params,
                )
                return optax.apply_updates(pre_params, updates)

            # VJP gives: ∇_{w_t}L_val(w_t, alpha) * B_t
            _, vjp_fn = jax.vjp(phi, hyperparams)
            local_hypergrad = vjp_fn(weighted_val_grad_wt)[0]

            hypergrad = jtu.tree_map(
                lambda a, b: a + b,
                hypergrad,
                local_hypergrad,
            )

        return hypergrad

    @partial(jax.jit, static_argnums=(0, 4, 5))
    def step(
        self,
        state: BilevelState,
        train_batch: DataBatch,
        val_batch: DataBatch,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
    ) -> BilevelState:
        """Perform one greedy optimization step.

        Algorithm
        ---------
        1. Unroll `unroll_steps` inner updates on train loss.
        2. Compute greedy hypergradient as a weighted sum of local greedy terms.
        3. Update hyperparameters using the outer optimizer.
        4. Keep the unrolled parameters / inner optimizer state.
        """
        final_params, final_inner_opt_state, last_train_loss, transitions = self._rollout(
            state=state,
            train_batch=train_batch,
            train_loss_fn=train_loss_fn,
        )

        hypergrad = self._compute_hypergradient_from_rollout(
            hyperparams=state.hyperparams,
            final_params=final_params,
            transitions=transitions,
            train_batch=train_batch,
            val_batch=val_batch,
            train_loss_fn=train_loss_fn,
            val_loss_fn=val_loss_fn,
        )

        outer_updates, new_outer_opt_state = self.outer_optimizer.update(
            hypergrad,
            state.outer_opt_state,
            state.hyperparams,
        )
        new_hyperparams = optax.apply_updates(state.hyperparams, outer_updates)

        val_loss = val_loss_fn(final_params, state.hyperparams, val_batch)

        metadata = {
            "train_loss": last_train_loss,
            "val_loss": val_loss,
            "hypergrad_norm": optax.global_norm(hypergrad),
            "param_norm": optax.global_norm(final_params),
            "hyperparam_norm": optax.global_norm(new_hyperparams),
        }

        return state.update(
            params=final_params,
            hyperparams=new_hyperparams,
            inner_opt_state=final_inner_opt_state,
            outer_opt_state=new_outer_opt_state,
            metadata=metadata,
        )

    def compute_hypergradient(
        self,
        state: BilevelState,
        train_batch: DataBatch,
        val_batch: DataBatch,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
    ) -> PyTree:
        """Compute hypergradient by unrolling inner optimization."""
        final_params, _, _, transitions = self._rollout(
            state=state,
            train_batch=train_batch,
            train_loss_fn=train_loss_fn,
        )

        return self._compute_hypergradient_from_rollout(
            hyperparams=state.hyperparams,
            final_params=final_params,
            transitions=transitions,
            train_batch=train_batch,
            val_batch=val_batch,
            train_loss_fn=train_loss_fn,
            val_loss_fn=val_loss_fn,
        )
