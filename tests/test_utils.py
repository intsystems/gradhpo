"""Tests for pytree / VJP helpers in ``gradhpo.utils.gradients``."""

import jax
import jax.numpy as jnp
import pytest

from gradhpo.utils.gradients import (
    tree_dot,
    tree_l2_norm,
    tree_lerp,
    tree_normalize,
    tree_zeros_like,
    update_w_star,
    vjp_wrt_both,
    vjp_wrt_lambda,
)


class TestTreeL2Norm:
    def test_simple_3_4_5(self):
        assert jnp.isclose(tree_l2_norm({'a': jnp.array([3.0, 4.0])}), 5.0)

    def test_empty_tree_returns_zero(self):
        assert float(tree_l2_norm({})) == 0.0

    def test_nested_pytree(self):
        tree = {'a': jnp.array([1.0]), 'b': {'c': jnp.array([2.0, 2.0])}}
        # sqrt(1 + 4 + 4) = 3
        assert jnp.isclose(tree_l2_norm(tree), 3.0)


class TestTreeNormalize:
    def test_unit_norm(self):
        normed = tree_normalize({'a': jnp.array([3.0, 4.0])})
        assert jnp.isclose(tree_l2_norm(normed), 1.0, atol=1e-6)

    def test_zero_tree_does_not_blow_up(self):
        normed = tree_normalize({'a': jnp.zeros(3)})
        assert jnp.all(jnp.isfinite(normed['a']))


class TestTreeOps:
    def test_tree_dot(self):
        a = {'x': jnp.array([1.0, 2.0]), 'y': jnp.array([3.0])}
        b = {'x': jnp.array([4.0, 5.0]), 'y': jnp.array([6.0])}
        assert jnp.isclose(tree_dot(a, b), 1*4 + 2*5 + 3*6)

    def test_tree_zeros_like(self):
        z = tree_zeros_like({'a': jnp.ones((2, 3))})
        assert jnp.allclose(z['a'], 0.0)
        assert z['a'].shape == (2, 3)

    @pytest.mark.parametrize("t,expected", [(0.0, 0.0), (0.5, 5.0), (1.0, 10.0)])
    def test_tree_lerp(self, t, expected):
        a = {'v': jnp.array([0.0])}
        b = {'v': jnp.array([10.0])}
        assert jnp.isclose(tree_lerp(a, b, t)['v'][0], expected)


class TestVJPHelpers:
    def test_vjp_wrt_lambda_matches_autodiff(self):
        def update(w, lam, batch):
            del batch
            return jax.tree.map(lambda x, l: x * jnp.exp(l), w, lam)

        w = {'p': jnp.array([1.0, 2.0])}
        lam = {'p': jnp.array([0.1, -0.2])}
        alpha = {'p': jnp.array([0.5, 1.5])}

        result = vjp_wrt_lambda(update, w, lam, batch=None, alpha=alpha)

        # d/dlam (w * exp(lam)) = w * exp(lam); contracted with alpha:
        expected = w['p'] * jnp.exp(lam['p']) * alpha['p']
        assert jnp.allclose(result['p'], expected, atol=1e-5)

    def test_vjp_wrt_both_returns_two_pytrees(self):
        def update(w, lam, batch):
            del batch
            return jax.tree.map(lambda x, l: x + l, w, lam)

        w = {'p': jnp.array([1.0, 2.0])}
        lam = {'p': jnp.array([0.5, -0.5])}
        alpha = {'p': jnp.array([1.0, 1.0])}

        a_w, a_l = vjp_wrt_both(update, w, lam, batch=None, alpha=alpha)
        # Both partial derivatives are identity, so the contraction equals alpha.
        assert jnp.allclose(a_w['p'], alpha['p'])
        assert jnp.allclose(a_l['p'], alpha['p'])


class TestUpdateWStar:
    def test_t1_returns_w_prev(self):
        w_star = {'a': jnp.array([99.0])}
        w_prev = {'a': jnp.array([1.0])}
        result = update_w_star(w_star, w_prev, gamma=0.99, t=1)
        assert jnp.allclose(result['a'], w_prev['a'])

    def test_t2_ema(self):
        w_star = {'a': jnp.array([1.0])}
        w_prev = {'a': jnp.array([2.0])}
        result = update_w_star(w_star, w_prev, gamma=0.5, t=2)
        # p_t = (0.5 - 0.25) / (1 - 0.25) = 1/3
        expected = (1.0 / 3) * 1.0 + (2.0 / 3) * 2.0
        assert jnp.isclose(result['a'][0], expected, atol=1e-5)

    def test_gamma0_returns_w_prev(self):
        result = update_w_star(
            {'a': jnp.array([1.0])},
            {'a': jnp.array([5.0])},
            gamma=0.0, t=3,
        )
        assert jnp.isclose(result['a'][0], 5.0, atol=1e-5)
