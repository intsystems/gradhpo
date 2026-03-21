# GradHPO: a JAX library for gradient-based hyperparameter optimization

If you have ever tuned hyperparameters in deep learning or meta-learning, you know the usual trade-off: either rely on expensive black-box search, or try to differentiate through training and run straight into memory and compute limits. In principle, gradient-based hyperparameter optimization is extremely appealing. In practice, exact hypergradients are often too costly because they require backpropagation through the entire inner optimization trajectory.

In this blog post, I will look at **GradHPO** — a **JAX-native** library designed to make gradient-based hyperparameter optimization more practical. Instead of focusing on a single “perfect” method, GradHPO brings together several hypergradient approximation techniques under one interface, making it easier to experiment with bilevel optimization without fully unrolling training.

## Why hypergradients are hard

Many machine learning problems can be written as bilevel optimization. At the lower level, we train model parameters. At the upper level, we optimize hyperparameters using validation performance. This setup appears naturally in regularization tuning, meta-learning, neural architecture search, and other settings where we want to optimize not just the model, but the training process itself.

The difficulty is that the outer objective depends on the result of the inner optimization. If we want exact gradients with respect to hyperparameters, we need to account for how the whole training trajectory changes. That quickly becomes expensive in both time and memory, especially for large models or long training loops.

This is exactly the gap GradHPO tries to close: it provides practical approximations of hypergradients that preserve much of the benefit of bilevel optimization while avoiding the full cost of exact differentiation.

## What GradHPO includes

The main idea behind GradHPO is simple: different applications need different trade-offs between gradient quality, runtime, and memory usage. That is why the library does not commit to a single algorithm. Instead, it currently implements three approaches.

**HyperDistill** is aimed at online hyperparameter updates and constant-cost hypergradient estimation. It is attractive when you want to update hyperparameters during training without storing or replaying the entire optimization history.

**T1–T2 + DARTS** is the lightest and simplest option. It uses a short-horizon, one-step lookahead style approximation, which makes it cheap and easy to use when fast experimentation matters more than accuracy of the hypergradient.

**Generalized Greedy** sits somewhere in between. It tries to incorporate information from the whole trajectory while still avoiding the memory cost of full unrolling. In other words, it offers a more informed approximation than short-horizon methods, but remains much more practical than exact approaches.

Together, these methods give users a spectrum of options rather than a single rigid recipe.

## A common interface for bilevel optimization

One of the most useful aspects of GradHPO is its design. The library exposes a shared abstraction through `BilevelOptimizer` and a unified optimizer state through `BilevelState`, while the specific algorithms are implemented as interchangeable optimizers.

This may sound like a small engineering detail, but in practice it matters a lot. It means you can switch from one hypergradient approximation strategy to another without rewriting the entire training pipeline. The surrounding bilevel optimization workflow stays the same, while only the hypergradient computation changes.

That makes GradHPO not just a collection of research implementations, but a usable experimental framework for comparing methods in a consistent way.

## Why JAX is a natural fit

GradHPO is built around **JAX**, and that choice makes a lot of sense. JAX provides exactly the tools you want for this kind of work: automatic differentiation with `grad`, vector-Jacobian products through `vjp`, compilation with `jit`, vectorization with `vmap`, and a flexible pytree-based way of handling structured parameters.

For optimizer transformations, the library relies on **Optax**, which keeps the optimization stack clean and composable. As a result, GradHPO fits naturally into modern JAX-based research workflows instead of feeling like an isolated prototype.

## When GradHPO is useful

GradHPO is particularly relevant when:

- you want to optimize hyperparameters with gradient information rather than black-box search,
- full differentiation through the inner loop is too expensive,
- you work on bilevel problems such as regularization tuning, meta-learning, or neural architecture search,
- you want to compare several hypergradient approximations in a unified framework.

In these settings, the library gives you practical tools to explore the design space between speed, memory efficiency, and gradient quality.

## Final thoughts

GradHPO is best understood as an attempt to turn gradient-based hyperparameter optimization from a beautiful but expensive idea into a practical research and engineering tool. If you need a simple and cheap baseline, you can start with **T1–T2 + DARTS**. If online updates and constant per-step cost matter most, **HyperDistill** becomes attractive. If you want to use richer trajectory information without paying for full unrolling, **Generalized Greedy** offers a middle ground.

That is the real strength of the library: not one universally best method, but a single platform for experimenting with several meaningful approaches to hypergradient estimation.

If you work with bilevel optimization in JAX, GradHPO looks like a promising foundation for both research prototypes and practical experimentation.
