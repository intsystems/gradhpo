"""OnlineHypergradientOptimizer: HyperDistill algorithm (Lee et al., ICLR 2022).

Implements online hyperparameter meta-learning with hypergradient distillation.
"""

from typing import Any, Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax

from mylib.core.base import BilevelOptimizer
from mylib.core.state import BilevelState
from mylib.core.types import PyTree, LossFn
from mylib.utils.gradients import (
    tree_zeros_like,
    tree_l2_norm,
    tree_normalize,
    tree_dot,
    tree_lerp,
    vjp_wrt_lambda,
    vjp_wrt_both,
    update_w_star,
)


class OnlineHypergradientOptimizer(BilevelOptimizer):
    """Online optimization with hypergradient distillation.

    Implements Algorithm 3 and Algorithm 4 from Lee et al. (ICLR 2022).

    Attributes:
        inner_optimizer: Optax optimizer for parameters (optional if update_fn given).
        outer_optimizer: Optax optimizer for hyperparameters (optional, falls back to SGD).
        gamma: EMA decay factor in [0, 1].
        estimation_period: Re-estimate theta every N episodes.
        T: Number of inner steps per episode.
        update_fn: Optional custom inner step function Phi(w, lam, batch) -> w_new.
    """

    def __init__(
        self,
        inner_optimizer: Optional[optax.GradientTransformation] = None,
        outer_optimizer: Optional[optax.GradientTransformation] = None,
        gamma: float = 0.99,
        estimation_period: int = 50,
        T: int = 20,
        update_fn: Optional[Callable] = None,
    ):
        """Initialize Online Hypergradient optimizer.

        Args:
            inner_optimizer: Optax optimizer for parameters.
            outer_optimizer: Optax optimizer for hyperparameters.
            gamma: EMA decay factor.
            estimation_period: Re-estimate theta every N episodes.
            T: Inner steps per episode.
            update_fn: Custom inner step Phi(w, lam, batch) -> w_new.
                       If provided, inner_optimizer is not used.
        """
        super().__init__(inner_optimizer, outer_optimizer)
        self.gamma = gamma
        self.estimation_period = estimation_period
        self.T = T
        self._update_fn = update_fn

    def _get_inner_step_fn(
        self, state: BilevelState, train_loss_fn: LossFn,
    ) -> Callable:
        """Return the inner step function Phi(w, lam, batch) -> w_new.

        If a custom update_fn was provided, use it directly.
        Otherwise, build one from train_loss_fn + inner_optimizer.
        """
        if self._update_fn is not None:
            return self._update_fn

        inner_opt = self.inner_optimizer
        inner_opt_state = state.inner_opt_state

        def phi(params, hyperparams, batch):
            grads = jax.grad(train_loss_fn, argnums=0)(params, hyperparams, batch)
            updates, _ = inner_opt.update(grads, inner_opt_state, params)
            return optax.apply_updates(params, updates)

        return phi

    def init(
        self,
        params: PyTree,
        hyperparams: PyTree,
    ) -> BilevelState:
        """Initialize state with hypergradient memory.

        Args:
            params: Initial model parameters.
            hyperparams: Initial hyperparameters.

        Returns:
            Initial BilevelState.
        """
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
            metadata={
                "w_star": params,
                "theta": 1.0,
                "phi": params,
            },
        )

    def compute_hypergradient(
        self,
        state: BilevelState,
        train_batch: Any,
        val_batch: Any,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
    ) -> PyTree:
        """Compute hypergradient w.r.t. hyperparameters.

        Uses the HyperDistill approximation: g = g_FO + theta * pi_t * v_t.

        Args:
            state: Current state (must have w_star, theta in metadata).
            train_batch: Training batch.
            val_batch: Validation batch.
            train_loss_fn: Training loss function.
            val_loss_fn: Validation loss function.

        Returns:
            Hypergradient with same structure as hyperparams.
        """
        t = state.step
        w_star = state.get_metric("w_star")
        theta = state.get_metric("theta", 1.0)

        phi_fn = self._get_inner_step_fn(state, train_loss_fn)

        # w_t = Phi(w_{t-1}, lam; D_t)
        w_new = phi_fn(state.params, state.hyperparams, train_batch)

        # alpha_t = dL_val(w_t, lam) / dw_t
        alpha = jax.grad(val_loss_fn, argnums=0)(
            w_new, state.hyperparams, val_batch)

        # g_FO = dL_val / dlam (direct gradient)
        g_fo = jax.grad(val_loss_fn, argnums=1)(
            w_new, state.hyperparams, val_batch)

        # v_t = alpha @ dPhi(w_star, lam; D_t) / dlam
        w_star_new = update_w_star(w_star, state.params, self.gamma, t)
        v_t = vjp_wrt_lambda(phi_fn, w_star_new, state.hyperparams,
                             train_batch, alpha)

        # scale = theta * (1 - gamma^t) / (1 - gamma)
        scale = theta * (1.0 - self.gamma ** t) / (1.0 - self.gamma + 1e-10)

        # g = g_FO + scale * v_t
        return jax.tree.map(lambda fo, v: fo + scale * v, g_fo, v_t)

    def step(
        self,
        state: BilevelState,
        train_batch: Any,
        val_batch: Any,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
        lr_hyper: Optional[float] = None,
    ) -> BilevelState:
        """Perform one online optimization step with distillation.

        Args:
            state: Current bilevel state.
            train_batch: Training data batch.
            val_batch: Validation data batch.
            train_loss_fn: Training loss function.
            val_loss_fn: Validation loss function.
            lr_hyper: Manual learning rate (used when outer_optimizer is None).

        Returns:
            Updated state.
        """
        t = state.step + 1
        w_star = state.get_metric("w_star")
        theta = state.get_metric("theta", 1.0)

        phi_fn = self._get_inner_step_fn(state, train_loss_fn)

        # 1. Update w_star (Eq. 13)
        w_star_new = update_w_star(w_star, state.params, self.gamma, t)

        # 2. Inner step: w_t = Phi(w_{t-1}, lam; D_t)
        w_new = phi_fn(state.params, state.hyperparams, train_batch)

        # 3. alpha_t and g_FO
        alpha = jax.grad(val_loss_fn, argnums=0)(
            w_new, state.hyperparams, val_batch)
        g_fo = jax.grad(val_loss_fn, argnums=1)(
            w_new, state.hyperparams, val_batch)

        # 4. v_t = alpha @ dPhi(w_star, lam; D_t) / dlam
        v_t = vjp_wrt_lambda(phi_fn, w_star_new, state.hyperparams,
                             train_batch, alpha)

        # 5. scale = theta * (1 - gamma^t) / (1 - gamma)
        scale = theta * (1.0 - self.gamma ** t) / (1.0 - self.gamma + 1e-10)

        # 6. Hypergradient
        hyper_grad = jax.tree.map(lambda fo, v: fo + scale * v, g_fo, v_t)

        # 7. Update hyperparams
        if self.outer_optimizer is not None:
            updates, new_outer_opt_state = self.outer_optimizer.update(
                hyper_grad, state.outer_opt_state, state.hyperparams)
            lam_new = optax.apply_updates(state.hyperparams, updates)
        else:
            assert lr_hyper is not None, "lr_hyper required when outer_optimizer is None"
            lam_new = jax.tree.map(
                lambda l, g: l - lr_hyper * g, state.hyperparams, hyper_grad)
            new_outer_opt_state = state.outer_opt_state

        # 8. Update inner opt state (if using optax inner optimizer)
        new_inner_opt_state = state.inner_opt_state
        if self.inner_optimizer is not None and self._update_fn is None:
            grads = jax.grad(train_loss_fn, argnums=0)(
                state.params, state.hyperparams, train_batch)
            _, new_inner_opt_state = self.inner_optimizer.update(
                grads, state.inner_opt_state, state.params)

        return state.update(
            params=w_new,
            hyperparams=lam_new,
            inner_opt_state=new_inner_opt_state,
            outer_opt_state=new_outer_opt_state,
            step=t,
            metadata={
                "w_star": w_star_new,
                "theta": theta,
                "phi": state.get_metric("phi"),
            },
        )

    def estimate_theta(
        self,
        state: BilevelState,
        train_batches: List[Any],
        val_batch: Any,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
    ) -> float:
        """Algorithm 4: estimate the scaling parameter theta.

        Runs a forward pass (T inner steps), then a DrMAD-style backward pass
        to collect samples (x_s, y_s), and fits theta = (x^T y) / (x^T x).

        Args:
            state: Current state (uses params and hyperparams).
            train_batches: List of T training batches.
            val_batch: Validation batch.
            train_loss_fn: Training loss function.
            val_loss_fn: Validation loss function.

        Returns:
            theta: Estimated linear scaling parameter.
        """
        phi_fn = self._get_inner_step_fn(state, train_loss_fn)
        w_init = state.params
        lam = state.hyperparams
        T = len(train_batches)
        gamma = self.gamma

        # --- Forward pass: w_0, w_1, ..., w_T ---
        weights = [w_init]
        w = w_init
        for t in range(T):
            w = phi_fn(w, lam, train_batches[t])
            weights.append(w)

        w_0, w_T = weights[0], weights[T]

        # alpha_T = dL_val(w_T, lam) / dw_T
        alpha_T = jax.grad(val_loss_fn, argnums=0)(w_T, lam, val_batch)

        alpha = alpha_T
        g_so = tree_zeros_like(lam)

        xs, ys = [], []
        w_star_est = None

        # --- Backward pass (DrMAD) ---
        for t_back in range(T, 0, -1):
            frac = (t_back - 1) / T
            w_hat = tree_lerp(w_0, w_T, frac)

            alpha_A, alpha_B = vjp_wrt_both(
                phi_fn, w_hat, lam, train_batches[t_back - 1], alpha)

            g_so = jax.tree.map(lambda a, b: a + b, g_so, alpha_B)
            alpha = alpha_A

            s = T - t_back + 1

            # Distilled w_s*
            if s == 1:
                w_star_est = weights[T - 1]
            else:
                gamma_s = gamma ** s
                p_s = (1.0 - gamma ** (s - 1)) / (1.0 - gamma_s + 1e-10)
                w_star_est = jax.tree.map(
                    lambda ws, w: p_s * ws + (1.0 - p_s) * w,
                    w_star_est, weights[T - s])

            v_s = vjp_wrt_lambda(phi_fn, w_star_est, lam,
                                 train_batches[T - s], alpha_T)

            v_norm = tree_l2_norm(v_s)
            x_s = v_norm * (1.0 - gamma ** s) / (1.0 - gamma + 1e-10)

            v_normalized = tree_normalize(v_s)
            y_s = tree_dot(v_normalized, g_so)

            xs.append(float(x_s))
            ys.append(float(y_s))

        x = jnp.array(xs)
        y = jnp.array(ys)
        theta = float(jnp.dot(x, y) / (jnp.dot(x, x) + 1e-10))
        return theta

    def run(
        self,
        state: BilevelState,
        M: int,
        get_train_batch: Callable,
        get_val_batch: Callable,
        train_loss_fn: LossFn,
        val_loss_fn: LossFn,
        lr_reptile: float = 1.0,
        lr_hyper: Optional[float] = None,
        callback: Optional[Callable] = None,
    ) -> BilevelState:
        """Full HyperDistill training loop (Algorithm 3).

        Runs M episodes, each with T inner steps and a Reptile update.

        Args:
            state: Initial state (from init()).
            M: Number of outer episodes.
            get_train_batch: Callable returning a training batch.
            get_val_batch: Callable returning a validation batch.
            train_loss_fn: Training loss function.
            val_loss_fn: Validation loss function.
            lr_reptile: Reptile learning rate for weight initialization.
            lr_hyper: Manual hyper LR (used when outer_optimizer is None).
            callback: Optional callback(episode, state).

        Returns:
            Final BilevelState.
        """
        phi = state.params

        for m in range(1, M + 1):
            # Re-estimate theta periodically (Algorithm 4)
            if (m - 1) % self.estimation_period == 0:
                est_batches = [get_train_batch() for _ in range(self.T)]
                est_val = get_val_batch()
                theta = self.estimate_theta(
                    state, est_batches, est_val, train_loss_fn, val_loss_fn)
                state = state.update(metadata={"theta": theta})

            # Reset step counter and params for new episode
            state = state.update(
                params=phi,
                step=0,
                metadata={"w_star": phi, "phi": phi},
            )

            lr_current = (
                lr_hyper * (1.0 - (m - 1) / M)
                if lr_hyper is not None
                else None
            )

            # Inner optimization (T steps)
            for t in range(1, self.T + 1):
                train_batch = get_train_batch()
                val_batch = get_val_batch()
                state = self.step(
                    state, train_batch, val_batch,
                    train_loss_fn, val_loss_fn, lr_current)

            # Reptile update: phi <- phi - lr_reptile * (phi - w_T)
            phi = jax.tree.map(
                lambda p, wt: p - lr_reptile * (p - wt), phi, state.params)

            if callback is not None:
                val_b = get_val_batch()
                val_loss = float(
                    val_loss_fn(state.params, state.hyperparams, val_b))
                state = state.update(
                    metadata={"val_loss": val_loss},
                )
                callback(m, state)

        state = state.update(params=phi, metadata={"phi": phi})
        return state
