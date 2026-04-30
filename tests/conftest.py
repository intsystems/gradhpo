"""Shared pytest fixtures and helpers for the gradhpo test suite.

Centralises a tiny synthetic classification problem and helper builders used
across most algorithm tests so individual test files stay focused on the
behaviour under test.
"""

import jax
import jax.numpy as jnp
import pytest


def _init_model(key):
    """Initialise a small 2-layer MLP for use in tests."""
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
    """Plain (params, batch) -> scalar cross-entropy loss."""
    x, y = batch
    logits = _forward(params, x)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * y, axis=-1))


def _bilevel_loss(params, hyperparams, batch):
    """(params, hyperparams, batch) -> scalar; ignores hyperparams."""
    return _loss_fn(params, batch)


def _bilevel_loss_with_l2(params, hyperparams, batch):
    """Bilevel loss with an explicit dependence on hyperparameters via L2.

    The coefficient and a separate quadratic term in ``hyperparams`` are
    chosen so the hypergradient is well above float32 round-off and the
    outer optimiser produces a visibly different update each step.
    """
    base = _loss_fn(params, batch)
    reg = sum(
        jnp.sum(jax.nn.softplus(h) * jnp.square(p))
        for p, h in zip(jax.tree.leaves(params), jax.tree.leaves(hyperparams))
    )
    # Explicit quadratic in hyperparams — guarantees a non-trivial direct
    # gradient term ``dL_val/dlam`` even when params are near zero.
    direct = sum(jnp.sum(jnp.square(h)) for h in jax.tree.leaves(hyperparams))
    return base + 1.0 * reg + 1e-3 * direct


def _update_fn(w, lr_params, batch):
    """Custom inner step: SGD with per-parameter softplus-rescaled LR."""
    grads = jax.grad(_loss_fn)(w, batch)
    return jax.tree.map(
        lambda w_i, lr_i, g_i: w_i - jax.nn.softplus(lr_i) * g_i,
        w, lr_params, grads,
    )


def _make_batch(key, n=32):
    """Random classification batch of size n with 4 features and 3 classes."""
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (n, 4))
    labels = jax.random.randint(k2, (n,), 0, 3)
    y = jax.nn.one_hot(labels, 3)
    return (x, y)


@pytest.fixture
def rng_key():
    """Fixed PRNGKey for reproducibility."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def model(rng_key):
    return _init_model(rng_key)


@pytest.fixture
def hyperparams(model):
    """Per-leaf log-LR initialised to softplus(-2.0) ≈ 0.127."""
    return jax.tree.map(lambda p: jnp.full_like(p, -2.0), model)


@pytest.fixture
def batch(rng_key):
    return _make_batch(rng_key)


@pytest.fixture
def loss_fns():
    """Pair of (train, val) bilevel loss functions (independent of hyperparams)."""
    return _bilevel_loss, _bilevel_loss


@pytest.fixture
def loss_fns_with_l2():
    """Bilevel losses that explicitly depend on hyperparameters."""
    return _bilevel_loss_with_l2, _bilevel_loss_with_l2


@pytest.fixture
def update_fn():
    return _update_fn


@pytest.fixture
def make_batch():
    return _make_batch


@pytest.fixture
def batch_iter(rng_key):
    """Deterministic infinite generator of random batches."""
    counter = {'i': 0}

    def _next():
        counter['i'] += 1
        return _make_batch(jax.random.fold_in(rng_key, counter['i']))

    return _next
