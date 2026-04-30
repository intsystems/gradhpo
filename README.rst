|test| |codecov| |docs| |pypi| |license|

.. |test| image:: https://github.com/intsystems/gradhpo/workflows/test/badge.svg
    :target: https://github.com/intsystems/gradhpo/actions/workflows/test.yml
    :alt: Test status

.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/gradhpo/master
    :target: https://app.codecov.io/gh/intsystems/gradhpo
    :alt: Test coverage

.. |docs| image:: https://github.com/intsystems/gradhpo/workflows/docs/badge.svg
    :target: https://intsystems.github.io/gradhpo/
    :alt: Docs status

.. |pypi| image:: https://img.shields.io/pypi/v/gradhpo.svg
    :target: https://pypi.org/project/gradhpo/
    :alt: PyPI version

.. |license| image:: https://img.shields.io/pypi/l/gradhpo.svg
    :target: https://github.com/intsystems/gradhpo/blob/master/LICENSE
    :alt: License


.. class:: center

    :Research Topic: Short-Horizon Gradient-Based Hyperparameter Optimization
    :Type of Work: Research Project
    :Authors: Eynullayev Altay, Rubtsov Denis, Karpeev Gleb

Abstract
========

Hyperparameter optimization is a fundamental challenge in modern machine learning, requiring
the selection of suitable hyperparameters given a validation dataset. Gradient-based methods
address this via bilevel optimization, enabling optimization over billion-dimensional search
spaces - far beyond the reach of classical approaches such as grid search or Bayesian
optimization. This project implements and wraps key gradient-based HPO algorithms as a
reusable JAX library: T1-T2 with DARTS numerical approximation, Generalized Greedy
Gradient-Based HPO, Online HPO with Hypergradient Distillation. The library provides a unified API suitable for
a broad class of tasks, with full documentation and automated testing.

Library Planning
================
Can be found `here <https://github.com/intsystems/gradhpo/tree/master/doc/Library_planning.pdf>`_.

Technical Report
================
Draft version can be found `here <https://github.com/intsystems/gradhpo/tree/master/doc/tech_report.md>`_.

Installation
============
The package is published on PyPI:

.. code-block:: bash

    pip install gradhpo

Alternatively, install from a source checkout:

.. code-block:: bash

    git clone https://github.com/intsystems/gradhpo.git
    pip install ./gradhpo/src

Software modules developed as part of the study
======================================================
1. A python package ``gradhpo`` published on `PyPI <https://pypi.org/project/gradhpo/>`_; sources `here <https://github.com/intsystems/gradhpo/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.com/intsystems/gradhpo/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/gradhpo/blob/master/code/main.ipynb>`_.
3. Documentation hosted at `intsystems.github.io/gradhpo <https://intsystems.github.io/gradhpo/>`_.
