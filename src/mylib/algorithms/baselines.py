"""Baseline bilevel optimizers: first-order and one-step lookahead."""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax

from mylib.core.base import BilevelOptimizer
from mylib.core.state import BilevelState
from mylib.core.types import PyTree, LossFn
from mylib.utils.gradients import (
    tree_zeros_like,
    vjp_wrt_lambda,
)


class FOOptimizer(BilevelOptimizer):
    """First-order baseline: ignores the second-order term entirely.

    Only uses g_FO = dL_val/dlam (direct gradient). If val_loss does not
    depend on hyperparams, no update is made to lambda.

    Attributes:
        inner_optimizer: Optax optimizer for parameters (optional if update_fn given).
        outer_optimizer: Optax optimizer for hyperparameters (optional).
        update_fn: Optional custom inner step Phi(w, lam, batch) -> w_new.
    """

    def __init__(
        self,
        inner_optimizer: Optional[optax.GradientTransformation] = None,
        outer_optimizer: Optional[optax.GradientTransformation] = None,
        update_fn: Optional[Callable] = None,
    ):
        super().__init__(inner_optimizer, outer_optimizer)
        self._update_fn = update_fn

    def _get_inner_step_fn(
        self, state: BilevelState, train_loss_fn: LossFn,
    ) -> Callable:
        if self._update_fn is not None:
            return self._update_fn
        inner_opt = self.inner_optimizer
        inner_opt_state = state.inner_opt_state

        def phi(params, hyperparams, batch):
            grads = jax.grad(train_loss_fn, argnums=0)(params, hyperparams, batch)
            updates, _ = inner_opt.update(grads, inner_opt_state, params)
            return optax.apply_updates(params, updates)

        return phi

    def init(self, params: PyTree, hyperparams: PyTree) -> BilevelState:
        inner_opt_state = (
            self.inner_optimizer.init(params)
            if self.inner_optimizer is not None
            else None
        )
        outer_opt_state = (
            self.outer_optimizer.init(hyperparams)
            if self.outer_optimizer is not None
            else None
        )
        return BilevelState(
            params=params,
            hyperparams=hyperparams,
            inner_opt_state=inner_opt_state,
            outer_opt_state=outer_opt_state,
            step=0,
            metadata={},
        )

    def compute_hypergradient(
        self,
        state: BilevelState,
        train_batch: Any,
        val_batch: Any,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
    ) -> PyTree:
        """First-order hypergradient: g_FO = dL_val/dlam."""
        phi_fn = self._get_inner_step_fn(state, train_loss_fn)
        w_new = phi_fn(state.params, state.hyperparams, train_batch)
        return jax.grad(val_loss_fn, argnums=1)(
            w_new, state.hyperparams, val_batch)

    def step(
        self,
        state: BilevelState,
        train_batch: Any,
        val_batch: Any,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
        lr_hyper: Optional[float] = None,
    ) -> BilevelState:
        phi_fn = self._get_inner_step_fn(state, train_loss_fn)
        w_new = phi_fn(state.params, state.hyperparams, train_batch)

        g_fo = jax.grad(val_loss_fn, argnums=1)(
            w_new, state.hyperparams, val_batch)

        if self.outer_optimizer is not None:
            updates, new_outer_state = self.outer_optimizer.update(
                g_fo, state.outer_opt_state, state.hyperparams)
            lam_new = optax.apply_updates(state.hyperparams, updates)
        else:
            assert lr_hyper is not None
            lam_new = jax.tree.map(
                lambda l, g: l - lr_hyper * g, state.hyperparams, g_fo)
            new_outer_state = state.outer_opt_state

        return state.update(
            params=w_new,
            hyperparams=lam_new,
            outer_opt_state=new_outer_state,
            step=state.step + 1,
        )

    def run(
        self,
        state: BilevelState,
        M: int,
        T: int,
        get_train_batch: Callable,
        get_val_batch: Callable,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
        lr_reptile: float = 1.0,
        lr_hyper: Optional[float] = None,
        callback: Optional[Callable] = None,
    ) -> BilevelState:
        phi = state.params

        for m in range(1, M + 1):
            state = state.update(params=phi, step=0)
            lr_current = (
                lr_hyper * (1.0 - (m - 1) / M) if lr_hyper is not None else None
            )

            for t in range(1, T + 1):
                train_batch = get_train_batch()
                val_batch = get_val_batch()
                state = self.step(
                    state, train_batch, val_batch,
                    train_loss_fn, val_loss_fn, lr_current)

            phi = jax.tree.map(
                lambda p, wt: p - lr_reptile * (p - wt), phi, state.params)

            if callback is not None:
                val_b = get_val_batch()
                val_loss = float(
                    val_loss_fn(state.params, state.hyperparams, val_b))
                state = state.update(metadata={"val_loss": val_loss})
                callback(m, state)

        state = state.update(params=phi)
        return state


class OneStepOptimizer(BilevelOptimizer):
    """One-step lookahead baseline (Luketina et al., 2016).

    Computes hypergradient using only the last step (short horizon, gamma=0).

    Attributes:
        inner_optimizer: Optax optimizer for parameters (optional if update_fn given).
        outer_optimizer: Optax optimizer for hyperparameters (optional).
        update_fn: Optional custom inner step Phi(w, lam, batch) -> w_new.
    """

    def __init__(
        self,
        inner_optimizer: Optional[optax.GradientTransformation] = None,
        outer_optimizer: Optional[optax.GradientTransformation] = None,
        update_fn: Optional[Callable] = None,
    ):
        super().__init__(inner_optimizer, outer_optimizer)
        self._update_fn = update_fn

    def _get_inner_step_fn(
        self, state: BilevelState, train_loss_fn: LossFn,
    ) -> Callable:
        if self._update_fn is not None:
            return self._update_fn
        inner_opt = self.inner_optimizer
        inner_opt_state = state.inner_opt_state

        def phi(params, hyperparams, batch):
            grads = jax.grad(train_loss_fn, argnums=0)(params, hyperparams, batch)
            updates, _ = inner_opt.update(grads, inner_opt_state, params)
            return optax.apply_updates(params, updates)

        return phi

    def init(self, params: PyTree, hyperparams: PyTree) -> BilevelState:
        inner_opt_state = (
            self.inner_optimizer.init(params)
            if self.inner_optimizer is not None
            else None
        )
        outer_opt_state = (
            self.outer_optimizer.init(hyperparams)
            if self.outer_optimizer is not None
            else None
        )
        return BilevelState(
            params=params,
            hyperparams=hyperparams,
            inner_opt_state=inner_opt_state,
            outer_opt_state=outer_opt_state,
            step=0,
            metadata={},
        )

    def compute_hypergradient(
        self,
        state: BilevelState,
        train_batch: Any,
        val_batch: Any,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
    ) -> PyTree:
        """One-step hypergradient: g_FO + alpha @ B_t."""
        phi_fn = self._get_inner_step_fn(state, train_loss_fn)
        w_new = phi_fn(state.params, state.hyperparams, train_batch)

        alpha = jax.grad(val_loss_fn, argnums=0)(
            w_new, state.hyperparams, val_batch)
        g_fo = jax.grad(val_loss_fn, argnums=1)(
            w_new, state.hyperparams, val_batch)

        g_so = vjp_wrt_lambda(
            phi_fn, state.params, state.hyperparams, train_batch, alpha)

        return jax.tree.map(lambda fo, so: fo + so, g_fo, g_so)

    def step(
        self,
        state: BilevelState,
        train_batch: Any,
        val_batch: Any,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
        lr_hyper: Optional[float] = None,
    ) -> BilevelState:
        phi_fn = self._get_inner_step_fn(state, train_loss_fn)
        w_new = phi_fn(state.params, state.hyperparams, train_batch)

        alpha = jax.grad(val_loss_fn, argnums=0)(
            w_new, state.hyperparams, val_batch)
        g_fo = jax.grad(val_loss_fn, argnums=1)(
            w_new, state.hyperparams, val_batch)

        g_so = vjp_wrt_lambda(
            phi_fn, state.params, state.hyperparams, train_batch, alpha)

        hyper_grad = jax.tree.map(lambda fo, so: fo + so, g_fo, g_so)

        if self.outer_optimizer is not None:
            updates, new_outer_state = self.outer_optimizer.update(
                hyper_grad, state.outer_opt_state, state.hyperparams)
            lam_new = optax.apply_updates(state.hyperparams, updates)
        else:
            assert lr_hyper is not None
            lam_new = jax.tree.map(
                lambda l, g: l - lr_hyper * g, state.hyperparams, hyper_grad)
            new_outer_state = state.outer_opt_state

        return state.update(
            params=w_new,
            hyperparams=lam_new,
            outer_opt_state=new_outer_state,
            step=state.step + 1,
        )

    def run(
        self,
        state: BilevelState,
        M: int,
        T: int,
        get_train_batch: Callable,
        get_val_batch: Callable,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
        lr_reptile: float = 1.0,
        lr_hyper: Optional[float] = None,
        callback: Optional[Callable] = None,
    ) -> BilevelState:
        phi = state.params

        for m in range(1, M + 1):
            state = state.update(params=phi, step=0)
            lr_current = (
                lr_hyper * (1.0 - (m - 1) / M) if lr_hyper is not None else None
            )

            for t in range(1, T + 1):
                train_batch = get_train_batch()
                val_batch = get_val_batch()
                state = self.step(
                    state, train_batch, val_batch,
                    train_loss_fn, val_loss_fn, lr_current)

            phi = jax.tree.map(
                lambda p, wt: p - lr_reptile * (p - wt), phi, state.params)

            if callback is not None:
                val_b = get_val_batch()
                val_loss = float(
                    val_loss_fn(state.params, state.hyperparams, val_b))
                state = state.update(metadata={"val_loss": val_loss})
                callback(m, state)

        state = state.update(params=phi)
        return state
