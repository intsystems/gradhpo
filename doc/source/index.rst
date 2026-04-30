========================================================
GradHpO: Gradient-Based Hyperparameter Optimization
========================================================

**Short-horizon gradient-based hyperparameter optimization in JAX**

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   overview
   installation
   quick_start
   tutorial
   api/index
   references

Welcome to GradHpO
==================

GradHpO is a JAX library for gradient-based hyperparameter optimization
in bilevel optimization problems.  Five algorithms are implemented under
a unified :class:`~gradhpo.core.base.BilevelOptimizer` interface:

- **HyperDistill** --- online optimization with EMA weight distillation
  (Lee et al., ICLR 2022).
- **T1-T2 with DARTS approximation** --- classic approach using finite-difference
  hypergradient estimation (Luketina et al., 2016; Liu et al., 2018).
- **Greedy** --- generalized greedy method with inner-loop unrolling
  (Anonymous, ICLR 2025).
- **FO (First-Order)** --- first-order baseline using direct gradient only.
- **One-Step** --- one-step lookahead baseline (Luketina et al., 2016).

Key Features
~~~~~~~~~~~~

- **Unified API** --- all algorithms inherit from ``BilevelOptimizer`` with
  methods ``init``, ``step``, ``compute_hypergradient``, and ``run``.
- **Arbitrary pytrees** --- model parameters and hyperparameters can be any
  nested structure of JAX arrays.
- **JAX-compatible ``BilevelState``** --- the state is registered as a JAX
  pytree, allowing it to be passed through ``jax.jit`` and ``jax.vmap``.
- **JIT compilation** --- all five ``step()`` methods are decorated with
  ``@partial(jax.jit, static_argnums=(0, 4, 5, 6))``.
- **Optax compatibility** --- ``GreedyOptimizer`` accepts
  ``optax.GradientTransformation`` objects for inner and outer optimizers.
- **Custom step function** --- other algorithms accept an arbitrary
  ``update_fn(w, lam, batch) -> w_new`` callable.

Quick Start
~~~~~~~~~~~

.. code-block:: python

   import jax
   import optax
   from gradhpo import OnlineHypergradientOptimizer, BilevelState

   # Define update_fn: one SGD step with per-parameter learning rates
   def update_fn(w, lr_params, batch):
       grads = jax.grad(train_loss)(w, batch)
       return jax.tree.map(
           lambda w_i, lr_i, g_i: w_i - jax.nn.softplus(lr_i) * g_i,
           w, lr_params, grads,
       )

   opt = OnlineHypergradientOptimizer(
       update_fn=update_fn,
       gamma=0.99,
       estimation_period=10,
       T=20,
   )
   state = opt.init(w_init, lam_init)

   # Main training loop
   state = opt.run(
       state, M=60,
       get_train_batch=train_iter,
       get_val_batch=val_iter,
       train_loss_fn=loss_fn,
       val_loss_fn=loss_fn,
       lr_hyper=3e-3,
   )

Navigation
~~~~~~~~~~

- :doc:`overview` --- problem formulation and algorithm descriptions
- :doc:`installation` --- installation and dependencies
- :doc:`quick_start` --- minimal working example
- :doc:`tutorial` --- detailed example with visualization
- :doc:`api/index` --- API reference
- :doc:`references` --- bibliography
