========
Tutorial
========

This section walks through a detailed example: training a two-layer MLP
on a synthetic classification task with per-parameter learning rate tuning
via HyperDistill and baselines.

The full script is in ``code/demo_hyperdistill.py``.
An interactive notebook covering all five algorithms and their comparison
is in ``code/demo_methods.ipynb``.

Task Description
================

- **Model**: MLP 10 → 32 → 5 (Xavier initialization), 517 parameters.
- **Data**: 5 classes, Gaussian clusters around fixed centers.
  The training set contains 20% label noise; the validation set is clean.
- **Hyperparameter** :math:`\lambda`: a vector of 517 values ---
  one learning rate per model parameter, passed through ``softplus``
  to guarantee positivity.

.. math::

   w_{t+1} = w_t - \mathrm{softplus}(\lambda) \odot \nabla L_{\mathrm{train}}(w_t).

This choice of hyperparameter is interesting because :math:`\lambda` does
not appear directly in :math:`L_{\mathrm{val}}`, so
:math:`g_{\mathrm{FO}} = 0` and all signal comes from the second-order term.

Model Definition
================

.. code-block:: python

   import jax
   import jax.numpy as jnp

   def init_mlp(key, in_dim, hidden_dim, out_dim):
       k1, k2 = jax.random.split(key)
       return {
           'w1': jax.random.normal(k1, (in_dim, hidden_dim)) * jnp.sqrt(2.0 / in_dim),
           'b1': jnp.zeros(hidden_dim),
           'w2': jax.random.normal(k2, (hidden_dim, out_dim)) * jnp.sqrt(2.0 / hidden_dim),
           'b2': jnp.zeros(out_dim),
       }

   def mlp_forward(params, x):
       h = jax.nn.relu(x @ params['w1'] + params['b1'])
       return h @ params['w2'] + params['b2']

Loss Functions and Inner Step
=============================

.. code-block:: python

   def cross_entropy_loss(params, batch):
       x, y = batch
       logits = mlp_forward(params, x)
       log_probs = jax.nn.log_softmax(logits, axis=-1)
       return -jnp.mean(jnp.sum(log_probs * y, axis=-1))

   # Bilevel interface wrapper: (params, hyperparams, batch) -> scalar
   def bilevel_loss(params, hyperparams, batch):
       return cross_entropy_loss(params, batch)

   def make_update_fn(loss_fn):
       def update_fn(w, lr_params, batch):
           grads = jax.grad(loss_fn)(w, batch)
           return jax.tree.map(
               lambda w_i, lr_i, g_i: w_i - jax.nn.softplus(lr_i) * g_i,
               w, lr_params, grads,
           )
       return update_fn

Running HyperDistill
====================

.. code-block:: python

   from gradhpo.algorithms.online import OnlineHypergradientOptimizer

   update_fn = make_update_fn(cross_entropy_loss)

   opt = OnlineHypergradientOptimizer(
       update_fn=update_fn,
       gamma=0.99,
       estimation_period=10,
       T=20,
   )

   state = opt.init(w_init, lam_init)

   losses = []

   def callback(episode, state):
       loss, _ = evaluate(state.params, X_val, Y_val)
       losses.append(loss)

   state = opt.run(
       state, M=60,
       get_train_batch=train_iter,
       get_val_batch=val_iter,
       train_loss_fn=bilevel_loss,
       val_loss_fn=bilevel_loss,
       lr_reptile=1.0,
       lr_hyper=3e-3,
       callback=callback,
   )

Running T1T2 and Greedy
=======================

.. code-block:: python

   import optax
   from gradhpo.algorithms.t1t2 import T1T2Optimizer
   from gradhpo.algorithms.greedy import GreedyOptimizer

   # T1T2 uses the same update_fn
   t1t2_opt = T1T2Optimizer(update_fn=update_fn, gamma=0.9, T=20)
   t1t2_state = t1t2_opt.init(w_init, lam_init)
   t1t2_state = t1t2_opt.run(
       t1t2_state, M=60,
       get_train_batch=train_iter,
       get_val_batch=val_iter,
       train_loss_fn=bilevel_loss,
       val_loss_fn=bilevel_loss,
       lr_hyper=3e-3,
   )

   # GreedyOptimizer takes Optax optimizers
   greedy_opt = GreedyOptimizer(
       inner_optimizer=optax.sgd(0.01),
       outer_optimizer=optax.adam(3e-3),
       unroll_steps=5,
       gamma=0.9,
   )
   greedy_state = greedy_opt.init(w_init, lam_init)
   greedy_state = greedy_opt.run(
       greedy_state, M=60,
       get_train_batch=train_iter,
       get_val_batch=val_iter,
       train_loss_fn=bilevel_loss,
       val_loss_fn=bilevel_loss,
   )

Comparison with Baselines
==========================

The baselines use the same interface:

.. code-block:: python

   from gradhpo.algorithms.baselines import OneStepOptimizer, FOOptimizer

   onestep_opt = OneStepOptimizer(update_fn=update_fn)
   onestep_state = onestep_opt.init(w_init, lam_init)
   onestep_state = onestep_opt.run(
       onestep_state, M=60, T=20,
       get_train_batch=train_iter,
       get_val_batch=val_iter,
       train_loss_fn=bilevel_loss,
       val_loss_fn=bilevel_loss,
       lr_reptile=1.0,
       lr_hyper=3e-3,
   )

   fo_opt = FOOptimizer(update_fn=update_fn)
   fo_state = fo_opt.init(w_init, lam_init)
   fo_state = fo_opt.run(
       fo_state, M=60, T=20,
       get_train_batch=train_iter,
       get_val_batch=val_iter,
       train_loss_fn=bilevel_loss,
       val_loss_fn=bilevel_loss,
       lr_hyper=3e-3,
   )

Expected Results
================

With default parameters (M=60, T=20, gamma=0.99):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Method
     - last-5 avg loss
     - best loss
   * - Fixed LR
     - ~0.186
     - ~0.175
   * - FO
     - ~0.183
     - ~0.172
   * - One-Step
     - ~0.171
     - ~0.160
   * - T1T2
     - ~0.168
     - ~0.155
   * - HyperDistill
     - ~0.162
     - ~0.148
   * - Greedy
     - ~0.165
     - ~0.152

HyperDistill consistently achieves the best result because it incorporates
information from the entire training trajectory via EMA distillation,
while One-Step only sees the local effect of the last step.
T1T2 and Greedy occupy an intermediate position.

Visualization
=============

.. code-block:: bash

   python code/demo_hyperdistill.py --plot

This saves a convergence plot to ``figures/hyperdistill_poc.png``.
The plot shows that HyperDistill and One-Step converge faster than Fixed LR,
with HyperDistill reaching a lower validation loss.

For an interactive comparison of all five algorithms, open the notebook:

.. code-block:: bash

   jupyter notebook code/demo_methods.ipynb

The notebook contains six sections: Demo 1 (HyperDistill), Demo 2 (One-Step),
Demo 3 (FO), Demo 4 (T1T2), Demo 5 (Greedy), and Demo 6 --- a comparison
of all methods on a single plot.
