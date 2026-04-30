===========================================================
gradhpo — gradient-based hyperparameter optimization in JAX
===========================================================

|test| |lint| |docs| |pypi| |license|

.. |test| image:: https://github.com/intsystems/gradhpo/workflows/test/badge.svg
    :target: https://github.com/intsystems/gradhpo/actions/workflows/test.yml
    :alt: Test status

.. |lint| image:: https://github.com/intsystems/gradhpo/workflows/lint/badge.svg
    :target: https://github.com/intsystems/gradhpo/actions/workflows/lint.yml
    :alt: PEP-8 (flake8)

.. |docs| image:: https://github.com/intsystems/gradhpo/workflows/docs/badge.svg
    :target: https://intsystems.github.io/gradhpo/
    :alt: Docs status

.. |pypi| image:: https://img.shields.io/pypi/v/gradhpo.svg
    :target: https://pypi.org/project/gradhpo/
    :alt: PyPI version

.. |license| image:: https://img.shields.io/pypi/l/gradhpo.svg
    :target: https://github.com/intsystems/gradhpo/blob/master/LICENSE
    :alt: License

``gradhpo`` is a `JAX <https://jax.readthedocs.io>`_ library for
**short-horizon gradient-based hyperparameter optimization** via bilevel
optimization. It packages five algorithms behind a single
``BilevelOptimizer`` interface:

- **HyperDistill** — online HPO with EMA hypergradient distillation
  (Lee et al., ICLR 2022).
- **T1-T2 with DARTS** — T1-T2 with finite-difference (DARTS) approximation
  of the second-order term (Luketina et al., 2016; Liu et al., 2018).
- **Greedy** — generalized greedy gradient-based HPO with inner-loop
  unrolling.
- **FO** — first-order baseline that uses only the direct gradient
  ``dL_val/dλ``.
- **One-Step** — one-step lookahead baseline (HyperDistill with γ=0).

All ``step()`` methods are JIT-compiled and accept arbitrary JAX pytrees
for both parameters and hyperparameters, so the same code works for a
single learning rate, a per-parameter LR vector, or any other structured
hyperparameter.

Installation
============

.. code-block:: bash

    pip install gradhpo

Requires Python ≥ 3.9. JAX, optax, scikit-learn and the rest of the
runtime dependencies are pulled in automatically.

Source install:

.. code-block:: bash

    git clone https://github.com/intsystems/gradhpo.git
    pip install ./gradhpo/src

Editable / dev install (recommended for contributors):

.. code-block:: bash

    git clone https://github.com/intsystems/gradhpo.git
    cd gradhpo
    pip install -e ./src
    pip install pytest pytest-cov flake8

Quick start
===========

.. code-block:: python

    import jax
    import jax.numpy as jnp
    from gradhpo import OnlineHypergradientOptimizer

    def loss_fn(params, hyperparams, batch):
        x, y = batch
        pred = x @ params['w']
        mse = jnp.mean((pred - y) ** 2)
        reg = jax.nn.softplus(hyperparams['log_lam']) * jnp.sum(params['w'] ** 2)
        return mse + reg

    def update_fn(w, lam, batch):
        g = jax.grad(loss_fn)(w, lam, batch)
        return jax.tree.map(lambda p, gp: p - 0.01 * gp, w, g)

    opt = OnlineHypergradientOptimizer(
        update_fn=update_fn, gamma=0.99, estimation_period=10, T=20,
    )
    state = opt.init({'w': jnp.zeros(10)}, {'log_lam': jnp.array(0.0)})

    state = opt.run(
        state, M=30,
        get_train_batch=get_train, get_val_batch=get_val,
        train_loss_fn=loss_fn, val_loss_fn=loss_fn,
        lr_hyper=1e-3,
    )

The same interface works for ``T1T2Optimizer``, ``GreedyOptimizer``,
``FOOptimizer`` and ``OneStepOptimizer``. See the documentation for a
side-by-side comparison and a full notebook.

Documentation
=============

- Full docs: https://intsystems.github.io/gradhpo/
- API reference: ``BilevelOptimizer``, ``BilevelState``, all algorithms,
  pytree/VJP utilities.
- Tutorial with a 2-layer MLP and a per-parameter learning rate vector.

Project information
===================

- Source: https://github.com/intsystems/gradhpo
- Issue tracker: https://github.com/intsystems/gradhpo/issues
- License: MIT

Citation
========

If you use ``gradhpo`` in academic work, please cite::

    Eynullayev, A., Rubtsov, D., & Karpeev, G. (2026).
    gradhpo: Gradient-Based Hyperparameter Optimization.
    MIPT Intelligent Systems.
