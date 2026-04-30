============
Installation
============

From Source
===========

.. code-block:: bash

   git clone https://github.com/intsystems/GradHpO.git
   cd GradHpO
   pip install -e ./src

Dependencies
============

.. list-table::
   :header-rows: 1
   :widths: 30 20

   * - Library
     - Version
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

To build the documentation, additionally install:

.. code-block:: bash

   pip install -r doc/requirements.txt

Verifying the Installation
==========================

.. code-block:: python

   import gradhpo
   print(gradhpo.__version__)

   from gradhpo import (
       BilevelState,
       BilevelOptimizer,
       OnlineHypergradientOptimizer,
       T1T2Optimizer,
       GreedyOptimizer,
       FOOptimizer,
       OneStepOptimizer,
   )

Building the Documentation
==========================

.. code-block:: bash

   cd doc
   make html

The output will be in ``doc/build/html/index.html``.

Troubleshooting
===============

**ModuleNotFoundError: No module named 'gradhpo'**
   Make sure the package was installed in development mode:
   run ``pip install -e ./src`` from the project root.

**No GPU acceleration**
   JAX uses CPU by default.  To enable GPU, install ``jaxlib`` with CUDA
   support following the
   `JAX installation guide <https://github.com/google/jax#installation>`_.

**Documentation build errors**
   Reinstall the documentation dependencies:
   ``pip install --upgrade -r doc/requirements.txt``.
