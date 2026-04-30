========
Overview
========

Problem Formulation
===================

The bilevel optimization problem in the machine learning context is formulated
as a nested optimization problem:

.. math::

   \min_{\lambda}\; L_{\mathrm{val}}\bigl(w^*(\lambda),\,\lambda\bigr),
   \qquad
   w^*(\lambda) = \arg\min_{w}\; L_{\mathrm{train}}(w,\,\lambda),

where :math:`w` are the model parameters (inner level),
:math:`\lambda` are the hyperparameters (outer level),
and :math:`L_{\mathrm{train}}`, :math:`L_{\mathrm{val}}` are the training
and validation loss functions respectively.

In practice, the solution :math:`w^*(\lambda)` is not available in closed
form, so it is approximated by a finite number of optimization steps.
Denoting one step of the inner optimizer as :math:`\Phi(w, \lambda; D)`,
after :math:`T` steps we obtain :math:`w_T`, which depends on
:math:`\lambda` through the entire trajectory.

Hypergradient
=============

The full hypergradient :math:`\mathrm{d}L_{\mathrm{val}} / \mathrm{d}\lambda`
is decomposed via the chain rule:

.. math::

   \frac{\mathrm{d}L_{\mathrm{val}}}{\mathrm{d}\lambda}
   = \underbrace{\frac{\partial L_{\mathrm{val}}}{\partial \lambda}}_{g_{\mathrm{FO}}}
   + \sum_{t=1}^{T}
     \underbrace{\alpha_t \cdot
       \prod_{s=t+1}^{T} A_s}_{} \cdot B_t,

where :math:`\alpha_t = \nabla_{w_t} L_{\mathrm{val}}(w_t)`,
:math:`A_s = \partial \Phi / \partial w`, and
:math:`B_t = \partial \Phi / \partial \lambda`.

Computing the full sum requires backpropagation through all :math:`T` steps,
which is expensive in memory and time.  GradHpO implements several
short-horizon approximations of this sum.

Implemented Algorithms
======================

HyperDistill
-------------

The online method with hypergradient distillation (Lee et al., ICLR 2022)
approximates the full second-order term via an EMA point :math:`w^*_t`:

.. math::

   w^*_t = p_t \cdot w^*_{t-1} + (1 - p_t) \cdot w_{t-1},
   \qquad
   p_t = \frac{\gamma - \gamma^t}{1 - \gamma^t}.

The hypergradient at step :math:`t`:

.. math::

   g_t = g_{\mathrm{FO}} + \theta \cdot \frac{1 - \gamma^t}{1 - \gamma}
   \cdot v_t,

where :math:`v_t = \alpha_t \cdot \partial\Phi(w^*_t, \lambda) / \partial\lambda`,
and the scalar :math:`\theta` is estimated periodically via a DrMAD-style
backward pass (Algorithm 4 in the paper).

Class: :class:`~gradhpo.algorithms.online.OnlineHypergradientOptimizer`.

T1-T2 with DARTS
-----------------

The T1-T2 algorithm (Luketina et al., 2016) separates the parameter and
hyperparameter update steps.  In our implementation, :math:`B_t` is computed
using the DARTS finite-difference approximation (Liu et al., 2018):

.. math::

   B_t \approx
   \frac{\Phi(w, \lambda + \varepsilon e_i) - \Phi(w, \lambda - \varepsilon e_i)}
   {2\varepsilon}.

This avoids explicit differentiation through the optimizer.

Class: :class:`~gradhpo.algorithms.t1t2.T1T2Optimizer`.

Greedy
------

The generalized greedy approach (Anonymous, ICLR 2025) accounts for
:math:`T` steps with exponential decay:

.. math::

   \hat{d}_\lambda =
   \nabla_\lambda L_{\mathrm{val}}(w_T)
   + \sum_{t=1}^{T} \gamma^{T-t}\,
     \nabla_{w_t} L_{\mathrm{val}}(w_t) \cdot B_t.

The parameter :math:`\gamma \in (0, 1]` controls the contribution of early
steps.  Unlike other algorithms, ``GreedyOptimizer`` accepts
``inner_optimizer`` and ``outer_optimizer`` as ``optax.GradientTransformation``
objects instead of a custom ``update_fn``.

Class: :class:`~gradhpo.algorithms.greedy.GreedyOptimizer`.

Baselines
---------

- **FO (First-Order)**: uses only the direct gradient
  :math:`g_{\mathrm{FO}} = \partial L_{\mathrm{val}} / \partial \lambda`.
  If :math:`\lambda` does not appear directly in :math:`L_{\mathrm{val}}`,
  the update is zero.

- **One-Step**: accounts for :math:`B_t` only at the last step,
  :math:`g = g_{\mathrm{FO}} + \alpha_T \cdot B_T`.
  Equivalent to HyperDistill with :math:`\gamma = 0`.

Classes: :class:`~gradhpo.algorithms.baselines.FOOptimizer`,
:class:`~gradhpo.algorithms.baselines.OneStepOptimizer`.

Library Architecture
====================

All algorithms inherit from ``BilevelOptimizer`` and implement three methods:

.. code-block:: python

   class BilevelOptimizer(ABC):
       def init(self, params, hyperparams) -> BilevelState: ...
       def step(self, state, train_batch, val_batch,
                train_loss_fn, val_loss_fn, lr_hyper) -> BilevelState: ...
       def compute_hypergradient(self, state, train_batch, val_batch,
                                 train_loss_fn, val_loss_fn) -> PyTree: ...

.. note::

   The ``step()`` signature of ``GreedyOptimizer`` does not include
   ``lr_hyper`` (the outer optimizer step size is set via ``outer_optimizer``
   at construction time).  All other algorithms require ``lr_hyper`` as a
   mandatory argument.

The optimization state is stored in ``BilevelState`` --- an immutable
container with fields ``params``, ``hyperparams``, ``inner_opt_state``,
``outer_opt_state``, ``step``, and ``metadata``.

``BilevelState`` is registered as a JAX pytree via
``jax.tree_util.register_pytree_node``, allowing it to be passed directly
to ``jax.jit``, ``jax.vmap``, and other JAX transformations.

All five ``step()`` methods are decorated with
``@partial(jax.jit, static_argnums=(0, 4, 5, 6))``, where the static
arguments are ``self`` (0), ``train_loss_fn`` (4), ``val_loss_fn`` (5),
and ``lr_hyper`` (6).
