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

   from gradhpo import OnlineHypergradientOptimizer

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

Тот же интерфейс работает для всех алгоритмов.  Для ``FOOptimizer`` и
``OneStepOptimizer`` метод ``run()`` требует явного аргумента ``T``
(число внутренних шагов).  ``GreedyOptimizer`` принимает Optax-оптимизаторы
вместо ``update_fn`` и не требует ``lr_hyper`` в ``step()``/``run()``.

.. code-block:: python

   import optax
   from gradhpo import (
       OnlineHypergradientOptimizer,
       T1T2Optimizer,
       GreedyOptimizer,
       FOOptimizer,
       OneStepOptimizer,
   )

   # Алгоритмы на основе update_fn
   methods_update_fn = {
       'FO':         FOOptimizer(update_fn=update_fn),
       'One-Step':   OneStepOptimizer(update_fn=update_fn),
       'HyperDistill': OnlineHypergradientOptimizer(
                         update_fn=update_fn, gamma=0.99,
                         estimation_period=10, T=20),
       'T1T2':       T1T2Optimizer(update_fn=update_fn, gamma=0.9, T=20),
   }

   for name, opt in methods_update_fn.items():
       st = opt.init(w_init, lam_init)
       # FO и One-Step требуют явного T
       extra = {'T': 20} if name in ('FO', 'One-Step') else {}
       st = opt.run(
           st, M=30,
           get_train_batch=get_train,
           get_val_batch=get_val,
           train_loss_fn=loss_fn,
           val_loss_fn=loss_fn,
           lr_hyper=1e-3,
           **extra,
       )
       lam = jax.nn.softplus(st.hyperparams['log_lam'])
       print(f"{name}: lambda = {lam:.4f}")

   # GreedyOptimizer использует Optax-оптимизаторы
   greedy = GreedyOptimizer(
       inner_optimizer=optax.sgd(0.01),
       outer_optimizer=optax.adam(1e-3),
       unroll_steps=5,
       gamma=0.9,
   )
   gs = greedy.init(w_init, lam_init)
   gs = greedy.run(
       gs, M=30,
       get_train_batch=get_train,
       get_val_batch=get_val,
       train_loss_fn=loss_fn,
       val_loss_fn=loss_fn,
   )
   lam = jax.nn.softplus(gs.hyperparams['log_lam'])
   print(f"Greedy: lambda = {lam:.4f}")

Подробный пример с визуализацией результатов приведён
в :doc:`tutorial`.  Полное описание API --- в :doc:`api/index`.
