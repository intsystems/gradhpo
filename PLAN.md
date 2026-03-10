# Short-horrizon gradient-based hyperparameter optimization GradHPO backlog
## Goal
GradFPO implements algorithms of short-horrizon gradient-based hyperparameter optimization. Hyperparameter optimization is a problem of finding suitable hyperparameters given a validation (or sometimes test) dataset. In contrast to classical hyperparameter optimization methods, gradient-based methods allow the researchers to perform hyperparameter optimization over a billion-dimmension search space.

## Steps:

### 1. Preparation

- Read papers (3) about short-horizon gradient-based optimization algorithms.

- Find all the realizations of the methods on jax/torch. Including, complete libraries for SHGBO.

- Identify pros and cons, similarities/differencies between all the realizations.

- Decide, what is going to be present as experiment. Which metrics/datasets/problems are going to be used to draw a comparison with the existing realizations.

### 2. Library realization

- Decide on the structure of the library: which functions/classes/programming paradigms are going to be used.

- Realize all the algorithms according to the chosen structure.

- Write tests, covering all the algorithms.

- Realize benchmark (base experiment), using the library's functionality

- Realize computational experiment

### 3. Documentation, report and publication

- Write and set documentation via sphinx and make it public

- Write a blog-post with the description of the library, its functionality and computational experiment

- Write a thorough tech report with extensive theoretical description of the algorithms and their comparison with current realizations.

- Release completed version of the library via pypi and enable pip-install of it.
