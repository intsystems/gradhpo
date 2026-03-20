==========
Установка
==========

Из исходников
=============

.. code-block:: bash

   git clone https://github.com/intsystems/GradHpO.git
   cd GradHpO
   pip install -e ./src

Зависимости
===========

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Библиотека
     - Версия
   * - JAX
     - >= 0.4.0
   * - jaxlib
     - >= 0.4.0
   * - optax
     - >= 0.1.7
   * - chex
     - >= 0.1.8
   * - numpy
     - >= 1.24.0
   * - typing-extensions
     - >= 4.5.0

Для сборки документации дополнительно:

.. code-block:: bash

   pip install -r doc/requirements.txt

Проверка установки
==================

.. code-block:: python

   import mylib
   print(mylib.__version__)

   from mylib import (
       BilevelState,
       BilevelOptimizer,
       OnlineHypergradientOptimizer,
       T1T2Optimizer,
   )

Сборка документации
===================

.. code-block:: bash

   cd doc
   make html

Результат будет в ``doc/build/html/index.html``.

Возможные проблемы
==================

**ModuleNotFoundError: No module named 'mylib'**
   Убедитесь, что установка выполнена в режиме разработки:
   ``pip install -e ./src`` из корня проекта.

**Нет GPU-ускорения**
   JAX по умолчанию использует CPU.  Для GPU установите
   ``jaxlib`` с поддержкой CUDA согласно
   `инструкции JAX <https://github.com/google/jax#installation>`_.

**Ошибки при сборке документации**
   Переустановите зависимости:
   ``pip install --upgrade -r doc/requirements.txt``.
