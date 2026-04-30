=========
Туториал
=========

В этом разделе мы рассмотрим подробный пример: обучение двухслойного
MLP на синтетической задаче классификации с подбором поэлементного
learning rate через HyperDistill и бейзлайны.

Полный код примера находится в ``code/demo_hyperdistill.py``.
Интерактивный ноутбук со всеми пятью алгоритмами и их сравнением ---
в ``code/demo_methods.ipynb``.

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

Запуск T1T2 и Greedy
=====================

.. code-block:: python

   import optax
   from gradhpo.algorithms.t1t2 import T1T2Optimizer
   from gradhpo.algorithms.greedy import GreedyOptimizer

   # T1T2 использует тот же update_fn
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

   # GreedyOptimizer принимает Optax-оптимизаторы
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

Сравнение с бейзлайнами
========================

Аналогично запускаем бейзлайны с тем же интерфейсом:

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

HyperDistill стабильно показывает лучший результат, поскольку
учитывает информацию со всей траектории обучения через EMA-дистилляцию,
тогда как One-Step видит только локальный эффект последнего шага.
T1T2 и Greedy занимают промежуточное положение.

Визуализация
============

.. code-block:: bash

   python code/demo_hyperdistill.py --plot

Команда сохранит график сходимости в ``figures/hyperdistill_poc.png``.
На графике видно, что HyperDistill и One-Step сходятся быстрее Fixed LR,
при этом HyperDistill достигает меньшего значения валидационного лосса.

Для интерактивного сравнения всех пяти алгоритмов откройте ноутбук:

.. code-block:: bash

   jupyter notebook code/demo_methods.ipynb

Ноутбук содержит шесть секций: Demo 1 (HyperDistill), Demo 2 (One-Step),
Demo 3 (FO), Demo 4 (T1T2), Demo 5 (Greedy) и Demo 6 --- сравнение
всех методов на одном графике.
