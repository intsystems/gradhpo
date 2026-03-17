"""Utility functions for gradient computation and pytree operations."""

from mylib.utils.gradients import (
    tree_l2_norm,
    tree_normalize,
    tree_dot,
    tree_zeros_like,
    tree_lerp,
    vjp_wrt_lambda,
    vjp_wrt_both,
    update_w_star,
)
