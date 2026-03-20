=========
Туториал
=========

В этом разделе мы рассмотрим подробный пример: обучение двухслойного
MLP на синтетической задаче классификации с подбором поэлементного
learning rate через HyperDistill и бейзлайны.

Полный код примера находится в ``code/demo_hyperdistill.py``.

Описание задачи
===============

- **Модель**: MLP 10 → 32 → 5 (Xavier-инициализация), 517 параметров.
- **Данные**: 5 классов, гауссовы облака вокруг фиксированных центров.
  Обучающая выборка содержит 20% шума в метках, валидационная --- чистая.
- **Гиперпараметр** :math:`\lambda`: вектор из 517 значений ---
  по одному learning rate на каждый параметр модели.
  Подаётся через ``softplus``, чтобы гарантировать положительность.

.. math::

   w_{t+1} = w_t - \mathrm{softplus}(\lambda) \odot \nabla L_{\mathrm{train}}(w_t).

Такой выбор гиперпараметра интересен тем, что :math:`\lambda` не входит
в :math:`L_{\mathrm{val}}` напрямую, следовательно
:math:`g_{\mathrm{FO}} = 0` и весь сигнал идёт от second-order term.

Определение модели
==================

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

Функции потерь и внутренний шаг
===============================

.. code-block:: python

   def cross_entropy_loss(params, batch):
       x, y = batch
       logits = mlp_forward(params, x)
       log_probs = jax.nn.log_softmax(logits, axis=-1)
       return -jnp.mean(jnp.sum(log_probs * y, axis=-1))

   # Обёртка для билевел-интерфейса: (params, hyperparams, batch) -> scalar
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

Запуск HyperDistill
===================

.. code-block:: python

   from mylib.algorithms.online import OnlineHypergradientOptimizer

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

Сравнение с бейзлайнами
========================

Аналогично запускаем бейзлайны с тем же интерфейсом:

.. code-block:: python

   from mylib.algorithms.baselines import OneStepOptimizer

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

Ожидаемые результаты
=====================

При параметрах по умолчанию (M=60, T=20, gamma=0.99):

.. list-table::
   :header-rows: 1
   :widths: 30 25 25

   * - Метод
     - last-5 avg loss
     - best loss
   * - Fixed LR
     - ~0.186
     - ~0.175
   * - One-Step
     - ~0.171
     - ~0.160
   * - HyperDistill
     - ~0.162
     - ~0.148

HyperDistill стабильно показывает лучший результат, поскольку
учитывает информацию со всей траектории обучения через EMA-дистилляцию,
тогда как One-Step видит только локальный эффект последнего шага.

Визуализация
============

.. code-block:: bash

   python code/demo_hyperdistill.py --plot

Команда сохранит график сходимости в ``figures/hyperdistill_poc.png``.
На графике видно, что HyperDistill и One-Step сходятся быстрее Fixed LR,
при этом HyperDistill достигает меньшего значения валидационного лосса.
