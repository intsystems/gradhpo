# Short-horrizon gradient-based hyperparameter optimization GradHPO backlog
## Goal
GradFPO implements algorithms of short-horrizon gradient-based hyperparameter optimization. Hyperparameter optimization is a problem of finding suitable hyperparameters given a validation (or sometimes test) dataset. In contrast to classical hyperparameter optimization methods, gradient-based methods allow the researchers to perform hyperparameter optimization over a billion-dimmension search space.

## Steps:

### 1. Preparation

- Read papers (3) about short-horizon gradient-based optimization algorithms.

- Find all the realizations of the methods on jax/torch. Including, complete libraries for SHGBO.

  - Find complete libraries and analyze how algorithms interact with models on torch/jax.
  - Find realizations of unique algorithms and try to decide the same interface

- Identify pros and cons, similarities/differencies between all the realizations.

- Decide, what is going to be presented as experiment. Which metrics/datasets/problems are going to be used to draw a comparison with the existing realizations.

### 2. Library realization

- Decide on the structure of the library: which functions/classes/programming paradigms are going to be used.

- Realize all the algorithms according to the chosen structure.

- Write tests, covering all the algorithms.

  - **Algorithm unit tests**: for each algorithm verify one hypergradient step on a simple model. Compare against finite-difference approximation; assert error below a threshold.
  - **Convergence tests**: on a toy task run a full optimization loop and check that validation loss reaches an expected level within a fixed number of iterations.
  - **Interface tests**: verify each algorithm conforms to the unified API — same signatures, compatibility with jax modules, correct handling of continuous and discrete hyperparameters.
  - **Edge-case tests**: zero learning rate, empty batch, single inner-loop step, hyperparameters at boundary values. Assert graceful errors, no crashes.
  - **Reproducibility tests**: with a fixed seed, two identical runs must produce bitwise-equal results.
  - **Integration test (benchmark)**: end-to-end run of an algorithm on a standard benchmark, assert metrics fall within an expected range.

- Realize benchmark (base experiment), using the library's functionality

- Realize computational experiment

### 3. Documentation, report and publication

- Write and set documentation via sphinx and make it public

- Write a blog-post with the description of the library, its functionality and computational experiment

- Write a thorough tech report with extensive theoretical description of the algorithms and their comparison with current realizations.

- Release completed version of the library via pypi and enable pip-install of it.

## FirePokerTable:

Вот таблица в формате Markdown на основе предоставленного скриншота:

| Имя задачи | Оценка сложности задачи |
| :--- | :---: |
| Read papers | 1 |
| Find all the realizations of the methods on jax/torch. Including, complete libraries for SHGBO. | 1 |
| Identify pros and cons, similarities/differences between all the realizations. | 1/2 |
| Decide, what is going to be presented as experiment. Which metrics/datasets/problems are going to be used to draw a comparison with the existing realizations. | 2 |
| Decide on the structure of the library: which functions/classes/programming paradigms are going to be used. | 2 |
| Realize all the algorithms according to the chosen structure. | 1 |
| Write tests, covering all the algorithms. | 1 |
| Realize benchmark (base experiment), using the library's functionality | 1 |
| Realize computational experiment | 2 |
| Write and set documentation via sphinx and make it public | 2 |
| Write a blog-post with the description of the library, its functionality and computational experiment | 1 |
| Write a thorough tech report with extensive theoretical description of the algorithms and their comparison with current realizations. | 3 |
| Release completed version of the library via pypi and enable pip-install of it. | 1/2 |
