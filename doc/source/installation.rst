============
Installation
============

Requirements
============

- Python ≥ 3.9
- pip ≥ 20.0

From PyPI
=========

The package is published on `PyPI <https://pypi.org/project/gradhpo/>`_:

.. code-block:: bash

   pip install gradhpo

This installs ``gradhpo`` and pulls in JAX, optax and the rest of the runtime
dependencies.

From Source
===========

Clone and install from source:

.. code-block:: bash

   git clone https://github.com/intsystems/gradhpo.git
   pip install ./gradhpo/src

For development (editable install + test/lint extras):

.. code-block:: bash

   git clone https://github.com/intsystems/gradhpo.git
   cd gradhpo
   pip install -e ./src
   pip install pytest pytest-cov flake8

Dependencies
============

Pinned ranges from |reqs|:

.. |reqs| replace:: ``src/requirements.txt``

.. list-table::
   :header-rows: 1
   :widths: 30 30

   * - Library
     - Version
   * - JAX
     - ``>=0.4.20,<1.0``
   * - jaxlib
     - ``>=0.4.20,<1.0``
   * - optax
     - ``>=0.1.7,<1.0``
   * - chex
     - ``>=0.1.8,<1.0``
   * - numpy
     - ``>=1.24.0,<3.0``
   * - scipy
     - ``>=1.10.0,<2.0``
   * - scikit-learn
     - ``>=1.3.0,<2.0``
   * - typing-extensions
     - ``>=4.5.0,<5.0``

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

Running the Test Suite
======================

.. code-block:: bash

   pip install pytest pytest-cov
   pytest tests/ --cov=gradhpo --cov-report=term-missing

The full suite contains 76 tests and currently reaches 100 % statement
coverage.

Building the Documentation
==========================

.. code-block:: bash

   cd doc
   sphinx-build -W --keep-going -b html source build/html

The output is written to ``doc/build/html/index.html``.

Troubleshooting
===============

**ModuleNotFoundError: No module named 'gradhpo'**
   The package is not in the active Python environment.  Install it with
   ``pip install gradhpo`` (PyPI) or ``pip install -e ./src`` from a
   source checkout.

**No GPU acceleration**
   JAX uses CPU by default.  To enable GPU, install ``jaxlib`` with CUDA
   support following the `JAX installation guide
   <https://github.com/google/jax#installation>`_.

**Documentation build errors**
   Reinstall the documentation dependencies:
   ``pip install --upgrade -r doc/requirements.txt``.
