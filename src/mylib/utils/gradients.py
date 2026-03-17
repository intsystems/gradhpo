"""Gradient utilities: pytree operations, VJP helpers, and EMA updates."""

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

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
# EMA weight update
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
