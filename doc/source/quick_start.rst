==============
Быстрый старт
==============

Этот раздел показывает минимальный рабочий пример: обучение линейной
модели с подбором коэффициента регуляризации через HyperDistill.

Подготовка данных
=================

.. code-block:: python

   import jax
   import jax.numpy as jnp

   key = jax.random.PRNGKey(0)
   k1, k2 = jax.random.split(key)

   # Синтетические данные: 200 обучающих, 100 валидационных
   X_train = jax.random.normal(k1, (200, 10))
   y_train = jnp.sign(X_train @ jnp.ones(10))

   X_val = jax.random.normal(k2, (100, 10))
   y_val = jnp.sign(X_val @ jnp.ones(10))

Определение модели
==================

Функция потерь принимает три аргумента ``(params, hyperparams, batch)`` ---
это единый интерфейс для всех алгоритмов библиотеки.

.. code-block:: python

   def loss_fn(params, hyperparams, batch):
       """MSE с L2-регуляризацией, где lambda = softplus(hyperparams)."""
       X, y = batch
       pred = X @ params['w']
       mse = jnp.mean((pred - y) ** 2)
       reg = jax.nn.softplus(hyperparams['log_lam']) * jnp.sum(params['w'] ** 2)
       return mse + reg

Инициализация
=============

.. code-block:: python

   from mylib import OnlineHypergradientOptimizer

   w_init = {'w': jnp.zeros(10)}
   lam_init = {'log_lam': jnp.array(0.0)}

   def update_fn(w, lam, batch):
       grads = jax.grad(loss_fn)(w, lam, batch)
       return jax.tree.map(lambda p, g: p - 0.01 * g, w, grads)

   opt = OnlineHypergradientOptimizer(
       update_fn=update_fn,
       gamma=0.99,
       estimation_period=10,
       T=20,
   )

   state = opt.init(w_init, lam_init)

Обучение
========

.. code-block:: python

   def get_train():
       return (X_train, y_train)

   def get_val():
       return (X_val, y_val)

   state = opt.run(
       state, M=30,
       get_train_batch=get_train,
       get_val_batch=get_val,
       train_loss_fn=loss_fn,
       val_loss_fn=loss_fn,
       lr_hyper=1e-3,
   )

   print(f"lambda = {jax.nn.softplus(state.hyperparams['log_lam']):.4f}")

Сравнение нескольких методов
============================

Тот же интерфейс работает для всех алгоритмов:

.. code-block:: python

   from mylib import OneStepOptimizer, FOOptimizer
   from mylib.algorithms.t1t2 import T1T2Optimizer

   methods = {
       'FO':         FOOptimizer(update_fn=update_fn),
       'One-Step':   OneStepOptimizer(update_fn=update_fn),
       'HyperDistill': OnlineHypergradientOptimizer(
                         update_fn=update_fn, gamma=0.99,
                         estimation_period=10, T=20),
   }

   for name, opt in methods.items():
       st = opt.init(w_init, lam_init)
       # ... запуск обучения ...
       print(f"{name}: lambda = ...")

Подробный пример с визуализацией результатов приведён
в :doc:`tutorial`.  Полное описание API --- в :doc:`api/index`.
