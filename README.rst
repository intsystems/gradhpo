|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/gradhpo/workflows/test/badge.svg
    :target: https://github.com/intsystems/gradhpo/tree/master
    :alt: Test status

.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/gradhpo/master
    :target: https://app.codecov.io/gh/intsystems/gradhpo
    :alt: Test coverage

.. |docs| image:: https://github.com/intsystems/gradhpo/workflows/docs/badge.svg
    :target: https://intsystems.github.io/gradhpo/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Short-Horizon Gradient-Based Hyperparameter Optimization
    :Тип научной работы: Research Project
    :Авторы: Eynullayev Altay, Rubtsov Denis, Karpeev Gleb

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

Research publications
===============================
1.

Presentations at conferences on the topic of research
================================================
1.

Software modules developed as part of the study
======================================================
1. A python package gradhpo with all implementation `here <https://github.com/intsystems/gradhpo/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.com/intsystems/gradhpo/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/gradhpo/blob/master/code/main.ipynb>`_.
