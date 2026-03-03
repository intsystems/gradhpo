# Short-horrizon gradient-based hyperparameter optimization GradHPO backlog
## Goal
GradFPO implements algorithms of short-horrizon gradient-based hyperparameter optimization. Hyperparameter optimization is a problem of finding suitable hyperparameters given a validation (or sometimes test) dataset. In contrast to classical hyperparameter optimization methods, gradient-based methods allow the researchers to perform hyperparameter optimization over a billion-dimmension search space.

## Steps:
 - Research the four target algorithms and review existing JAX implementations to understand their mathematical foundations and computational patterns.

 - Design a unified, user-friendly API in JAX (with optional PyTorch support) for gradient-based hyperparameter optimizers, ensuring compatibility with common training loops.

 - Incrementally implement the algorithms from simplest to hardest—T1‑T2, Greedy, Online Meta‑Learning, Implicit Differentiation—reusing and refactoring existing code where possible.

 - Develop a comprehensive test suite covering unit tests for core components and integration tests on small models to guarantee numerical stability and correctness.

 - Benchmark all methods on diverse tasks (e.g., image classification, text classification) comparing accuracy, speed, and scalability, and create clear visualizations.

 - Write thorough documentation including API reference, tutorials, and a technical report, and publish a blog post to showcase the project’s value.

 - Prepare executable demo notebooks demonstrating real‑world usage and finalize the library for public release on GitHub.
