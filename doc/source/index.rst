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

GradHpO --- библиотека на JAX для градиентного подбора гиперпараметров
в задачах билевел-оптимизации.  Реализованы четыре алгоритма с единым
интерфейсом :class:`~mylib.core.base.BilevelOptimizer`:

- **T1-T2 с DARTS-аппроксимацией** --- классический подход с конечно-разностной
  оценкой гиперградиента.
- **Greedy** --- обобщённый жадный метод с развёрткой внутреннего цикла.
- **HyperDistill** --- онлайн-оптимизация с EMA-дистилляцией весов
  (Lee et al., ICLR 2022).
- **Бейзлайны** --- first-order (FO) и one-step lookahead (Luketina et al., 2016).

Ключевые особенности
~~~~~~~~~~~~~~~~~~~~

- **Единый API** --- все алгоритмы наследуют ``BilevelOptimizer`` с методами
  ``init``, ``step``, ``compute_hypergradient``.
- **Произвольные pytree** --- параметры модели и гиперпараметры могут быть
  любой вложенной структурой JAX-массивов.
- **Совместимость с Optax** --- внутренний и внешний оптимизаторы задаются
  через ``optax.GradientTransformation``.
- **Пользовательская функция шага** --- вместо optax-оптимизатора можно
  передать произвольную функцию ``Phi(w, lam, batch) -> w_new``.

Быстрый старт
~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import optax
   from mylib import OnlineHypergradientOptimizer, BilevelState

   # Задаём update_fn: один шаг SGD с поэлементным LR
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

   # Основной цикл
   state = opt.run(
       state, M=60,
       get_train_batch=train_iter,
       get_val_batch=val_iter,
       train_loss_fn=loss_fn,
       val_loss_fn=loss_fn,
       lr_hyper=3e-3,
   )

Навигация
~~~~~~~~~

- :doc:`overview` --- постановка задачи и описание алгоритмов
- :doc:`installation` --- установка и зависимости
- :doc:`quick_start` --- минимальный пример
- :doc:`tutorial` --- подробный пример с визуализацией
- :doc:`api/index` --- справочник по API
- :doc:`references` --- список литературы
