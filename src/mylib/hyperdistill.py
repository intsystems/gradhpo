"""HyperDistill: Online Hyperparameter Meta-Learning with Hypergradient Distillation.

Implementation of the algorithm from:
Lee et al., "Online Hyperparameter Meta-Learning with Hypergradient Distillation",
ICLR 2022.

This module provides:
- `hyperdistill_step`: one online HO step (Algorithm 3, inner loop)
- `linear_estimation`: estimate scaling parameter theta (Algorithm 4)
- `run_hyperdistill`: full HyperDistill training loop (Algorithm 3)
- `one_step_step`: one-step lookahead baseline (Luketina et al., 2016)
- `fo_step`: first-order baseline
- `run_baseline`: training loop for baselines
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional, Tuple, List

PyTree = Any


# ---------------------------------------------------------------------------
# Pytree utilities
# ---------------------------------------------------------------------------

def tree_l2_norm(tree: PyTree) -> jnp.ndarray:
    """L2 norm of a pytree."""
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.array(0.0)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))


def tree_normalize(tree: PyTree, eps: float = 1e-8) -> PyTree:
    """Normalize pytree to unit L2 norm."""
    norm = tree_l2_norm(tree)
    return jax.tree.map(lambda x: x / (norm + eps), tree)


def tree_dot(a: PyTree, b: PyTree) -> jnp.ndarray:
    """Inner product of two pytrees with the same structure."""
    products = jax.tree.map(lambda x, y: jnp.sum(x * y), a, b)
    return sum(jax.tree.leaves(products))


def tree_zeros_like(tree: PyTree) -> PyTree:
    """Create a pytree of zeros with the same structure."""
    return jax.tree.map(jnp.zeros_like, tree)


def tree_lerp(a: PyTree, b: PyTree, t: float) -> PyTree:
    """Linear interpolation: (1-t)*a + t*b."""
    return jax.tree.map(lambda x, y: (1.0 - t) * x + t * y, a, b)


# ---------------------------------------------------------------------------
# VJP helpers
# ---------------------------------------------------------------------------

def vjp_wrt_lambda(update_fn: Callable, w: PyTree, lam: PyTree,
                   batch: Any, alpha: PyTree) -> PyTree:
    """Compute alpha @ dPhi/dlambda via VJP."""
    def phi_lam(l):
        return update_fn(w, l, batch)
    _, vjp_fn = jax.vjp(phi_lam, lam)
    return vjp_fn(alpha)[0]


def vjp_wrt_both(update_fn: Callable, w: PyTree, lam: PyTree,
                 batch: Any, alpha: PyTree) -> Tuple[PyTree, PyTree]:
    """Compute alpha @ dPhi/dw and alpha @ dPhi/dlambda simultaneously."""
    def phi_wl(w_, l_):
        return update_fn(w_, l_, batch)
    _, vjp_fn = jax.vjp(phi_wl, w, lam)
    alpha_A, alpha_B = vjp_fn(alpha)
    return alpha_A, alpha_B


# ---------------------------------------------------------------------------
# Core HyperDistill (Algorithm 3 + 4)
# ---------------------------------------------------------------------------

def update_w_star(w_star: PyTree, w_prev: PyTree,
                  gamma: float, t: int) -> PyTree:
    """Eq. 13: sequential update of distilled weight point.

    t=1:  w_1* = w_0
    t>=2: w_t* = p_t * w_{t-1}* + (1 - p_t) * w_{t-1}
    where p_t = (gamma - gamma^t) / (1 - gamma^t).
    """
    if t <= 1:
        return w_prev
    gamma_t = gamma ** t
    p_t = (gamma - gamma_t) / (1.0 - gamma_t + 1e-10)
    return jax.tree.map(lambda ws, w: p_t * ws + (1.0 - p_t) * w,
                        w_star, w_prev)


def hyperdistill_step(
    w: PyTree,
    lam: PyTree,
    w_star: PyTree,
    theta: float,
    gamma: float,
    t: int,
    train_batch: Any,
    val_batch: Any,
    update_fn: Callable,
    loss_val_fn: Callable,
    lr_hyper: float,
    has_direct_grad: bool = False,
) -> Tuple[PyTree, PyTree, PyTree]:
    """One online HO step of HyperDistill (lines 8-13 of Algorithm 3).

    Args:
        w: current model weights (w_{t-1}).
        lam: current hyperparameters.
        w_star: distilled weight point (EMA) from previous step.
        theta: linear estimator parameter.
        gamma: decay factor in [0, 1].
        t: inner step index (1-indexed).
        train_batch: training mini-batch D_t.
        val_batch: validation mini-batch.
        update_fn: Phi(w, lam, batch) -> w_new.
        loss_val_fn: L_val(w, batch) -> scalar   (has_direct_grad=False)
                     L_val(w, lam, batch) -> scalar (has_direct_grad=True).
        lr_hyper: hyperparameter learning rate.
        has_direct_grad: whether L_val depends on lam directly.

    Returns:
        (w_new, lam_new, w_star_new).
    """
    # 1. Update w_star (Eq. 13) — uses w_{t-1}
    w_star_new = update_w_star(w_star, w, gamma, t)

    # 2. Inner SGD step: w_t = Phi(w_{t-1}, lam; D_t)
    w_new = update_fn(w, lam, train_batch)

    # 3. Compute alpha_t = dL_val(w_t) / dw_t  and  g_FO
    if has_direct_grad:
        alpha = jax.grad(loss_val_fn, argnums=0)(w_new, lam, val_batch)
        g_fo = jax.grad(loss_val_fn, argnums=1)(w_new, lam, val_batch)
    else:
        alpha = jax.grad(loss_val_fn, argnums=0)(w_new, val_batch)
        g_fo = tree_zeros_like(lam)

    # 4. Compute v_t = alpha_t @ dPhi(w_t*, lam; D_t*) / dlam
    #    For simplicity D_t* = D_t (proper subsampling omitted in POC).
    v_t = vjp_wrt_lambda(update_fn, w_star_new, lam, train_batch, alpha)

    # 5. pi_t* * f_t = theta * (1 - gamma^t) / (1 - gamma) * v_t
    scale = theta * (1.0 - gamma ** t) / (1.0 - gamma + 1e-10)

    # 6. Hypergradient g = g_FO + scale * v_t
    hyper_grad = jax.tree.map(lambda fo, v: fo + scale * v, g_fo, v_t)

    # 7. Update lambda
    lam_new = jax.tree.map(lambda l, g: l - lr_hyper * g, lam, hyper_grad)

    return w_new, lam_new, w_star_new


def linear_estimation(
    w_init: PyTree,
    lam: PyTree,
    gamma: float,
    train_batches: List[Any],
    val_batch: Any,
    update_fn: Callable,
    loss_val_fn: Callable,
    T: int,
    has_direct_grad: bool = False,
) -> float:
    """Algorithm 4: estimate the scaling parameter theta.

    Runs a forward pass (T inner steps), then a DrMAD-style backward pass
    to collect samples (x_s, y_s), and fits theta = (x^T y) / (x^T x).

    Args:
        w_init: initial weights (phi).
        lam: current hyperparameters.
        gamma: decay factor.
        train_batches: list of T training batches.
        val_batch: validation batch.
        update_fn: Phi(w, lam, batch) -> w_new.
        loss_val_fn: L_val(w, batch) -> scalar.
        T: number of inner steps.
        has_direct_grad: whether L_val depends on lam directly.

    Returns:
        theta: estimated linear scaling parameter.
    """
    # --- Forward pass: w_0, w_1, ..., w_T ---
    weights = [w_init]
    w = w_init
    for t in range(T):
        w = update_fn(w, lam, train_batches[t])
        weights.append(w)

    w_0, w_T = weights[0], weights[T]

    # alpha_T = dL_val(w_T) / dw_T  (fixed for the entire backward pass)
    if has_direct_grad:
        alpha_T = jax.grad(loss_val_fn, argnums=0)(w_T, lam, val_batch)
    else:
        alpha_T = jax.grad(loss_val_fn, argnums=0)(w_T, val_batch)

    alpha = alpha_T          # propagated backward
    g_so = tree_zeros_like(lam)  # accumulated second-order term

    xs, ys = [], []
    w_star_est = None

    # --- Backward pass (DrMAD) ---
    for t_back in range(T, 0, -1):
        # DrMAD: interpolated weight
        frac = (t_back - 1) / T
        w_hat = tree_lerp(w_0, w_T, frac)

        # alpha @ A_hat  and  alpha @ B_hat  (simultaneous VJP)
        alpha_A, alpha_B = vjp_wrt_both(
            update_fn, w_hat, lam, train_batches[t_back - 1], alpha)

        # Accumulate SO term: g^SO += alpha @ B_hat
        g_so = jax.tree.map(lambda a, b: a + b, g_so, alpha_B)

        # Propagate alpha backward: alpha <- alpha @ A_hat
        alpha = alpha_A

        # Horizon size s = T - t_back + 1
        s = T - t_back + 1

        # --- Distilled w_s* (Eq. 17) ---
        if s == 1:
            w_star_est = weights[T - 1]
        else:
            gamma_s = gamma ** s
            p_s = (1.0 - gamma ** (s - 1)) / (1.0 - gamma_s + 1e-10)
            w_star_est = jax.tree.map(
                lambda ws, w: p_s * ws + (1.0 - p_s) * w,
                w_star_est, weights[T - s])

        # v_s = alpha_T @ dPhi(w_s*, lam; D_s*) / dlam
        batch_star = train_batches[T - s]
        v_s = vjp_wrt_lambda(update_fn, w_star_est, lam, batch_star, alpha_T)

        # x_s = ||v_s|| * (1 - gamma^s) / (1 - gamma)
        v_norm = tree_l2_norm(v_s)
        x_s = v_norm * (1.0 - gamma ** s) / (1.0 - gamma + 1e-10)

        # y_s = sigma(v_s)^T g^SO
        v_normalized = tree_normalize(v_s)
        y_s = tree_dot(v_normalized, g_so)

        xs.append(float(x_s))
        ys.append(float(y_s))

    # theta = (x^T y) / (x^T x)
    x = jnp.array(xs)
    y = jnp.array(ys)
    theta = float(jnp.dot(x, y) / (jnp.dot(x, x) + 1e-10))
    return theta


def run_hyperdistill(
    w_init: PyTree,
    lam_init: PyTree,
    gamma: float,
    T: int,
    M: int,
    lr_hyper: float,
    lr_reptile: float,
    update_fn: Callable,
    loss_val_fn: Callable,
    get_train_batch: Callable,
    get_val_batch: Callable,
    estimation_period: int = 50,
    has_direct_grad: bool = False,
    callback: Optional[Callable] = None,
) -> Tuple[PyTree, PyTree]:
    """Algorithm 3: full HyperDistill training loop.

    Args:
        w_init: initial model weights.
        lam_init: initial hyperparameters.
        gamma: decay factor (e.g. 0.99).
        T: inner steps per episode.
        M: number of outer episodes.
        lr_hyper: hyperparameter learning rate.
        lr_reptile: Reptile learning rate for weight initialization.
        update_fn: Phi(w, lam, batch) -> w_new.
        loss_val_fn: L_val(w, batch) -> scalar.
        get_train_batch: callable returning a training batch.
        get_val_batch: callable returning a validation batch.
        estimation_period: re-estimate theta every N episodes.
        has_direct_grad: whether L_val depends on lam directly.
        callback: optional callback(episode, w, lam, metrics).

    Returns:
        (phi, lam): learned initialization and hyperparameters.
    """
    phi = w_init
    lam = lam_init
    theta = 1.0

    for m in range(1, M + 1):
        # Re-estimate theta periodically (Algorithm 4)
        if (m - 1) % estimation_period == 0:
            est_batches = [get_train_batch() for _ in range(T)]
            est_val = get_val_batch()
            theta = linear_estimation(
                phi, lam, gamma, est_batches, est_val,
                update_fn, loss_val_fn, T, has_direct_grad)

        # Inner optimisation
        w = phi
        w_star = phi
        lr_current = lr_hyper * (1.0 - (m - 1) / M)  # linear decay

        for t in range(1, T + 1):
            train_batch = get_train_batch()
            val_batch = get_val_batch()

            w, lam, w_star = hyperdistill_step(
                w, lam, w_star, theta, gamma, t,
                train_batch, val_batch,
                update_fn, loss_val_fn,
                lr_current, has_direct_grad)

        # Reptile update: phi <- phi - lr_reptile * (phi - w_T)
        phi = jax.tree.map(lambda p, wt: p - lr_reptile * (p - wt), phi, w)

        if callback is not None:
            val_b = get_val_batch()
            if has_direct_grad:
                val_loss = float(loss_val_fn(w, lam, val_b))
            else:
                val_loss = float(loss_val_fn(w, val_b))
            callback(m, w, lam, {'val_loss': val_loss, 'theta': theta})

    return phi, lam


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def fo_step(
    w: PyTree, lam: PyTree,
    train_batch: Any, val_batch: Any,
    update_fn: Callable, loss_val_fn: Callable,
    lr_hyper: float, has_direct_grad: bool = False,
) -> Tuple[PyTree, PyTree]:
    """First-order baseline: ignores the second-order term entirely."""
    w_new = update_fn(w, lam, train_batch)

    if has_direct_grad:
        g_fo = jax.grad(loss_val_fn, argnums=1)(w_new, lam, val_batch)
    else:
        g_fo = tree_zeros_like(lam)

    lam_new = jax.tree.map(lambda l, g: l - lr_hyper * g, lam, g_fo)
    return w_new, lam_new


def one_step_step(
    w: PyTree, lam: PyTree,
    train_batch: Any, val_batch: Any,
    update_fn: Callable, loss_val_fn: Callable,
    lr_hyper: float, has_direct_grad: bool = False,
) -> Tuple[PyTree, PyTree]:
    """One-step lookahead baseline (Luketina et al., 2016).

    Computes hypergradient using only the last step (short horizon, gamma=0).
    """
    w_new = update_fn(w, lam, train_batch)

    if has_direct_grad:
        alpha = jax.grad(loss_val_fn, argnums=0)(w_new, lam, val_batch)
        g_fo = jax.grad(loss_val_fn, argnums=1)(w_new, lam, val_batch)
    else:
        alpha = jax.grad(loss_val_fn, argnums=0)(w_new, val_batch)
        g_fo = tree_zeros_like(lam)

    # SO term: alpha @ B_t  (single JVP at current step only)
    g_so = vjp_wrt_lambda(update_fn, w, lam, train_batch, alpha)

    hyper_grad = jax.tree.map(lambda fo, so: fo + so, g_fo, g_so)
    lam_new = jax.tree.map(lambda l, g: l - lr_hyper * g, lam, hyper_grad)
    return w_new, lam_new


def run_baseline(
    w_init: PyTree,
    lam_init: PyTree,
    T: int,
    M: int,
    lr_hyper: float,
    lr_reptile: float,
    update_fn: Callable,
    loss_val_fn: Callable,
    get_train_batch: Callable,
    get_val_batch: Callable,
    method: str = 'fo',
    has_direct_grad: bool = False,
    callback: Optional[Callable] = None,
) -> Tuple[PyTree, PyTree]:
    """Training loop for FO or one-step baseline.

    Args:
        method: 'fo' or 'one_step'.
    """
    phi = w_init
    lam = lam_init
    step_fn = fo_step if method == 'fo' else one_step_step

    for m in range(1, M + 1):
        w = phi
        lr_current = lr_hyper * (1.0 - (m - 1) / M)

        for t in range(1, T + 1):
            train_batch = get_train_batch()
            val_batch = get_val_batch()
            w, lam = step_fn(
                w, lam, train_batch, val_batch,
                update_fn, loss_val_fn, lr_current, has_direct_grad)

        phi = jax.tree.map(lambda p, wt: p - lr_reptile * (p - wt), phi, w)

        if callback is not None:
            val_b = get_val_batch()
            if has_direct_grad:
                val_loss = float(loss_val_fn(w, lam, val_b))
            else:
                val_loss = float(loss_val_fn(w, val_b))
            callback(m, w, lam, {'val_loss': val_loss})

    return phi, lam
