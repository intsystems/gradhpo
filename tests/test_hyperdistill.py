"""Unit tests for the HyperDistill algorithm (OOP API)."""

import jax
import jax.numpy as jnp
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mylib.core.state import BilevelState
from mylib.core.types import DataBatch, LossFunctions, PyTree, LossFn
from mylib.algorithms.online import OnlineHypergradientOptimizer
from mylib.algorithms.baselines import FOOptimizer, OneStepOptimizer
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


# ---------------------------------------------------------------------------
# Helpers -- tiny model for tests
# ---------------------------------------------------------------------------

def _init_model(key):
    k1, k2 = jax.random.split(key)
    return {
        'w1': jax.random.normal(k1, (4, 8)) * 0.1,
        'b1': jnp.zeros(8),
        'w2': jax.random.normal(k2, (8, 3)) * 0.1,
        'b2': jnp.zeros(3),
    }


def _forward(params, x):
    h = jax.nn.relu(x @ params['w1'] + params['b1'])
    return h @ params['w2'] + params['b2']


def _loss_fn(params, batch):
    """Internal loss: (params, batch) -> scalar."""
    x, y = batch
    logits = _forward(params, x)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * y, axis=-1))


def _loss_fn_bilevel(params, hyperparams, batch):
    """Bilevel loss: (params, hyperparams, batch) -> scalar."""
    return _loss_fn(params, batch)


def _update_fn(w, lr_params, batch):
    grads = jax.grad(_loss_fn)(w, batch)
    return jax.tree.map(
        lambda w_i, lr_i, g_i: w_i - jax.nn.softplus(lr_i) * g_i,
        w, lr_params, grads)


def _make_batch(key, n=32):
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (n, 4))
    labels = jax.random.randint(k2, (n,), 0, 3)
    y = jax.nn.one_hot(labels, 3)
    return (x, y)


# ---------------------------------------------------------------------------
# Pytree utility tests
# ---------------------------------------------------------------------------

class TestTreeUtils:
    def test_tree_l2_norm(self):
        tree = {'a': jnp.array([3.0, 4.0])}
        assert jnp.isclose(tree_l2_norm(tree), 5.0)

    def test_tree_normalize(self):
        tree = {'a': jnp.array([3.0, 4.0])}
        normed = tree_normalize(tree)
        assert jnp.isclose(tree_l2_norm(normed), 1.0, atol=1e-6)

    def test_tree_dot(self):
        a = {'x': jnp.array([1.0, 2.0]), 'y': jnp.array([3.0])}
        b = {'x': jnp.array([4.0, 5.0]), 'y': jnp.array([6.0])}
        assert jnp.isclose(tree_dot(a, b), 1*4 + 2*5 + 3*6)

    def test_tree_zeros_like(self):
        tree = {'a': jnp.ones((2, 3))}
        z = tree_zeros_like(tree)
        assert jnp.allclose(z['a'], 0.0)
        assert z['a'].shape == (2, 3)

    def test_tree_lerp(self):
        a = {'v': jnp.array([0.0])}
        b = {'v': jnp.array([10.0])}
        mid = tree_lerp(a, b, 0.3)
        assert jnp.isclose(mid['v'][0], 3.0)


# ---------------------------------------------------------------------------
# VJP tests
# ---------------------------------------------------------------------------

class TestVJP:
    def test_vjp_wrt_lambda_nonzero(self):
        key = jax.random.PRNGKey(0)
        w = _init_model(key)
        lr = jax.tree.map(lambda p: jnp.zeros_like(p), w)
        batch = _make_batch(key)
        alpha = jax.grad(_loss_fn)(w, batch)

        result = vjp_wrt_lambda(_update_fn, w, lr, batch, alpha)
        norm = tree_l2_norm(result)
        assert float(norm) > 0, 'VJP should be non-zero'

    def test_vjp_wrt_both_consistent(self):
        key = jax.random.PRNGKey(1)
        w = _init_model(key)
        lr = jax.tree.map(lambda p: jnp.zeros_like(p), w)
        batch = _make_batch(key)
        alpha = jax.grad(_loss_fn)(w, batch)

        alpha_A, alpha_B = vjp_wrt_both(_update_fn, w, lr, batch, alpha)
        alpha_B_single = vjp_wrt_lambda(_update_fn, w, lr, batch, alpha)

        for leaf_a, leaf_b in zip(jax.tree.leaves(alpha_B),
                                  jax.tree.leaves(alpha_B_single)):
            assert jnp.allclose(leaf_a, leaf_b, atol=1e-5)


# ---------------------------------------------------------------------------
# w_star update tests
# ---------------------------------------------------------------------------

class TestWStarUpdate:
    def test_t1_returns_w_prev(self):
        w_star = {'a': jnp.array([99.0])}
        w_prev = {'a': jnp.array([1.0])}
        result = update_w_star(w_star, w_prev, gamma=0.99, t=1)
        assert jnp.allclose(result['a'], w_prev['a'])

    def test_t2_ema(self):
        w_star = {'a': jnp.array([1.0])}
        w_prev = {'a': jnp.array([2.0])}
        gamma = 0.5
        result = update_w_star(w_star, w_prev, gamma, t=2)
        expected = (1.0 / 3) * 1.0 + (2.0 / 3) * 2.0
        assert jnp.isclose(result['a'][0], expected, atol=1e-5)

    def test_gamma0_returns_w_prev(self):
        w_star = {'a': jnp.array([1.0])}
        w_prev = {'a': jnp.array([5.0])}
        result = update_w_star(w_star, w_prev, gamma=0.0, t=3)
        assert jnp.isclose(result['a'][0], 5.0, atol=1e-5)


# ---------------------------------------------------------------------------
# BilevelState tests
# ---------------------------------------------------------------------------

class TestBilevelState:
    def test_create(self):
        params = {'w': jnp.ones(3)}
        hyperparams = {'lr': jnp.ones(3) * 0.01}
        state = BilevelState.create(params, hyperparams, None, None)
        assert state.step == 0
        assert state.metadata == {}
        assert jnp.allclose(state.params['w'], 1.0)

    def test_update(self):
        state = BilevelState.create(
            {'w': jnp.ones(3)}, {'lr': jnp.ones(3)}, None, None)
        new_state = state.update(
            params={'w': jnp.zeros(3)},
            step=5,
            metadata={'val_loss': 0.5},
        )
        assert jnp.allclose(new_state.params['w'], 0.0)
        assert new_state.step == 5
        assert new_state.get_metric('val_loss') == 0.5
        assert jnp.allclose(state.params['w'], 1.0)

    def test_get_metric_default(self):
        state = BilevelState.create({'w': jnp.ones(1)}, {}, None, None)
        assert state.get_metric('missing', 42) == 42

    def test_metadata_merge(self):
        state = BilevelState(
            params={}, hyperparams={},
            inner_opt_state=None, outer_opt_state=None,
            step=0, metadata={'a': 1, 'b': 2},
        )
        new_state = state.update(metadata={'b': 99, 'c': 3})
        assert new_state.metadata == {'a': 1, 'b': 99, 'c': 3}


# ---------------------------------------------------------------------------
# Algorithm step tests
# ---------------------------------------------------------------------------

class TestAlgorithmSteps:
    def test_online_optimizer_step(self):
        key = jax.random.PRNGKey(42)
        w = _init_model(key)
        lr = jax.tree.map(lambda p: jnp.full_like(p, -2.0), w)
        batch = _make_batch(key)

        opt = OnlineHypergradientOptimizer(
            update_fn=_update_fn, gamma=0.99, T=5)
        state = opt.init(w, lr)
        new_state = opt.step(
            state, batch, batch,
            _loss_fn_bilevel, _loss_fn_bilevel, lr_hyper=1e-3)

        assert not jnp.allclose(new_state.params['w1'], w['w1'])
        assert new_state.step == 1

    def test_onestep_optimizer_step(self):
        key = jax.random.PRNGKey(42)
        w = _init_model(key)
        lr = jax.tree.map(lambda p: jnp.full_like(p, -2.0), w)
        batch = _make_batch(key)

        opt = OneStepOptimizer(update_fn=_update_fn)
        state = opt.init(w, lr)
        new_state = opt.step(
            state, batch, batch,
            _loss_fn_bilevel, _loss_fn_bilevel, lr_hyper=1e-3)

        assert not jnp.allclose(new_state.params['w1'], w['w1'])

    def test_fo_optimizer_no_lambda_change(self):
        key = jax.random.PRNGKey(42)
        w = _init_model(key)
        lr = jax.tree.map(lambda p: jnp.full_like(p, -2.0), w)
        batch = _make_batch(key)

        opt = FOOptimizer(update_fn=_update_fn)
        state = opt.init(w, lr)
        new_state = opt.step(
            state, batch, batch,
            _loss_fn_bilevel, _loss_fn_bilevel, lr_hyper=1e-3)

        for l_old, l_new in zip(jax.tree.leaves(lr),
                                jax.tree.leaves(new_state.hyperparams)):
            assert jnp.allclose(l_old, l_new)

    def test_compute_hypergradient(self):
        key = jax.random.PRNGKey(42)
        w = _init_model(key)
        lr = jax.tree.map(lambda p: jnp.full_like(p, -2.0), w)
        batch = _make_batch(key)

        opt = OnlineHypergradientOptimizer(
            update_fn=_update_fn, gamma=0.99, T=5)
        state = opt.init(w, lr)
        state = state.update(step=1)

        hg = opt.compute_hypergradient(
            state, batch, batch, _loss_fn_bilevel, _loss_fn_bilevel)

        assert tree_l2_norm(hg) > 0


# ---------------------------------------------------------------------------
# Linear estimation test
# ---------------------------------------------------------------------------

class TestLinearEstimation:
    def test_returns_finite_theta(self):
        key = jax.random.PRNGKey(7)
        w = _init_model(key)
        lr = jax.tree.map(lambda p: jnp.full_like(p, -2.0), w)
        T = 5
        batches = [_make_batch(jax.random.fold_in(key, i)) for i in range(T)]
        val_batch = _make_batch(jax.random.PRNGKey(99))

        opt = OnlineHypergradientOptimizer(
            update_fn=_update_fn, gamma=0.99, T=T)
        state = opt.init(w, lr)

        theta = opt.estimate_theta(
            state, batches, val_batch, _loss_fn_bilevel, _loss_fn_bilevel)

        assert jnp.isfinite(theta), f'theta should be finite, got {theta}'


# ---------------------------------------------------------------------------
# Reproducibility test
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_deterministic_with_same_seed(self):
        def _run(seed):
            key = jax.random.PRNGKey(seed)
            w = _init_model(key)
            lr = jax.tree.map(lambda p: jnp.full_like(p, -2.0), w)
            batch = _make_batch(key)

            opt = OnlineHypergradientOptimizer(
                update_fn=_update_fn, gamma=0.99, T=5)
            state = opt.init(w, lr)
            new_state = opt.step(
                state, batch, batch,
                _loss_fn_bilevel, _loss_fn_bilevel, lr_hyper=1e-3)
            return new_state.params, new_state.hyperparams

        w_a, lr_a = _run(42)
        w_b, lr_b = _run(42)

        for la, lb in zip(jax.tree.leaves(w_a), jax.tree.leaves(w_b)):
            assert jnp.allclose(la, lb)
        for la, lb in zip(jax.tree.leaves(lr_a), jax.tree.leaves(lr_b)):
            assert jnp.allclose(la, lb)


# ---------------------------------------------------------------------------
# Hypergradient quality test (finite-difference check)
# ---------------------------------------------------------------------------

class TestHypergradientQuality:
    def test_so_term_aligns_with_finite_diff(self):
        key = jax.random.PRNGKey(123)
        w = _init_model(key)
        lr = jax.tree.map(lambda p: jnp.full_like(p, -2.0), w)
        batch = _make_batch(key)

        # Analytical: one-step hypergradient SO term
        w_new = _update_fn(w, lr, batch)
        alpha = jax.grad(_loss_fn)(w_new, batch)
        so_analytical = vjp_wrt_lambda(_update_fn, w, lr, batch, alpha)

        # Autodiff through the whole pipeline for ground truth
        def val_after_step(lr_params):
            w_new = _update_fn(w, lr_params, batch)
            return _loss_fn(w_new, batch)

        fd_grad = jax.grad(val_after_step)(lr)

        so_flat = jnp.concatenate([l.ravel() for l in jax.tree.leaves(so_analytical)])
        fd_flat = jnp.concatenate([l.ravel() for l in jax.tree.leaves(fd_grad)])

        cos_sim = jnp.dot(so_flat, fd_flat) / (
            jnp.linalg.norm(so_flat) * jnp.linalg.norm(fd_flat) + 1e-10)
        assert float(cos_sim) > 0.9, \
            f'Cosine similarity between SO term and autodiff = {float(cos_sim):.4f}'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
