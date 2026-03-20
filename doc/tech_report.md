# GradHPO: A JAX Library for Gradient-Based Hyperparameter Optimization

**Technical Report — Draft**

---

## Abstract

Bilevel optimization is a foundational framework for hyperparameter optimization, meta-learning, and neural architecture search. Computing exact hypergradients is computationally prohibitive, motivating a family of approximation methods that trade off accuracy, memory, and scalability. This report presents **GradHPO** (`mylib`), a JAX-based Python library that provides unified, composable implementations of three gradient-based hyperparameter optimization algorithms: (1) **Online Hyperparameter Meta-Learning with Hypergradient Distillation** (HyperDistill), (2) the **T1–T2 method with numerical DARTS approximation**, and (3) **Generalized Greedy Gradient-based optimization**. We describe the theoretical foundations of each method, discuss the library's architecture and key design decisions, and outline the software stack.

---

## 1. Introduction

### 1.1 Problem Setting

Consider a model with parameters $w \in \mathbb{R}^P$ and hyperparameters $\lambda \in \mathbb{R}^H$. The inner-level optimization trains the model parameters on a training loss:

$$w_{t} = \Phi(w_{t-1}, \lambda; \mathcal{D}_t^{\text{train}}), \quad t = 1, \dots, T,$$

where $\Phi(\cdot, \cdot; \cdot)$ is a smooth update operator (e.g., a gradient descent step). The outer-level objective seeks hyperparameters that minimize a validation loss after $T$ inner steps:

$$\lambda^* = \arg\min_{\lambda} \; L_{\text{val}}(w_T, \lambda), \quad \text{s.t.} \quad w_t = \Phi(w_{t-1}, \lambda), \; t \in \overline{1, T}.$$

This is a bilevel optimization problem. The key challenge is computing the *hypergradient* $d_\lambda L_{\text{val}}(w_T, \lambda)$, which captures how the validation loss changes with respect to $\lambda$ through the entire inner optimization trajectory.

### 1.2 Hypergradient Decomposition

By the chain rule, the total derivative of the validation loss with respect to $\lambda$ decomposes as:

$$d_\lambda L_{\text{val}}(w_T, \lambda) = \underbrace{\nabla_\lambda L_{\text{val}}(w_T, \lambda)}_{g_{\text{FO}}} + \underbrace{\nabla_{w_T} L_{\text{val}}(w_T, \lambda) \cdot \frac{dw_T}{d\lambda}}_{g_{\text{SO}}},$$

where $g_{\text{FO}}$ is the *first-order* (direct) term and $g_{\text{SO}}$ is the *second-order* (indirect) term that accounts for the response of the learned parameters to changes in $\lambda$.

The response Jacobian $dw_T / d\lambda$ is computed via the recursive chain rule (Franceschi et al., 2017):

$$\frac{dw_T}{d\lambda} = \sum_{t=1}^{T} \left( \prod_{k=t+1}^{T} A_k \right) B_t, \quad A_k = \frac{\partial \Phi(w_{k-1}, \lambda)}{\partial w_{k-1}}, \quad B_t = \frac{\partial \Phi(w_{t-1}, \lambda)}{\partial \lambda}.$$

Computing this exactly requires either storing the full trajectory $w_1, \dots, w_T$ (Reverse-Mode Differentiation, memory $\mathcal{O}(PT)$) or performing $H$ forward passes (Forward-Mode Differentiation, time $\mathcal{O}(HT)$). Both are impractical for large-scale problems. The three methods implemented in GradHPO offer different approximation strategies to make hypergradient computation tractable.

### 1.3 Overview of Methods

| Property | HyperDistill | T1–T2 + DARTS | Generalized Greedy |
|---|---|---|---|
| Horizon | Full (online) | Single step | Full trajectory |
| Memory | $\mathcal{O}(P + H)$ | $\mathcal{O}(P + H)$ | $\mathcal{O}(P + H)$ |
| JVPs per outer step | 1 | 1 | $T$ |
| Inner convergence required | No | No | No |
| Scalable to large $H$ | Yes | Yes | Yes |

---

## 2. Method 1: Online Hyperparameter Meta-Learning with Hypergradient Distillation

### 2.1 Background

The HyperDistill algorithm was introduced by Lee et al. (2022) to enable *online* hyperparameter optimization — updating $\lambda$ at every inner step rather than waiting for the inner loop to complete. The key insight is to *distill* the expensive second-order term $g_{\text{SO}}$ into a single Jacobian-vector product (JVP) evaluated at a carefully constructed *distilled weight point*.

### 2.2 Hessian Identity Approximation

The main computational bottleneck in $g_{\text{SO}}$ is the product of Jacobians $\prod_{k=t+1}^{T} A_k$. HyperDistill approximates each $A_k$ with a scalar multiple of the identity:

$$A_k \approx \gamma I, \quad \gamma \in (0, 1),$$

which is motivated by the observation that for gradient descent on an $L$-smooth convex loss with step size $\eta \leq 1/L$, the spectrum of $A_k = I - \eta \nabla^2_w L_{\text{train}}$ lies in $[0, 1]$. Under this approximation, the product telescopes:

$$\prod_{k=t+1}^{T} A_k \approx \gamma^{T-t} I.$$

### 2.3 Distilled Weight Point

Substituting the Hessian identity approximation into the second-order term yields:

$$g_{\text{SO}} \approx \sum_{t=1}^{T} \gamma^{T-t} \nabla_{w_T} L_{\text{val}}(w_T, \lambda) \cdot B_t.$$

The key contribution of HyperDistill is recognizing that this weighted sum of JVPs can be *distilled* into a single JVP evaluated at a synthetic weight point $w_t^*$, constructed via an exponential moving average (EMA):

$$w_1^* = w_0, \quad w_t^* = p_t \, w_{t-1}^* + (1 - p_t) \, w_{t-1},$$

where the mixing coefficient is:

$$p_t = \frac{\gamma - \gamma^t}{1 - \gamma^t}.$$

This ensures that $w_t^*$ encodes the entire history of parameter updates, weighted by the geometric decay $\gamma^{T-t}$, while requiring only constant additional memory.

### 2.4 Linear Scaling Estimator

The distilled JVP $f_t(w_t^*, \mathcal{D}_t^*)$ approximates the *direction* of $g_{\text{SO}}$ but not its magnitude. HyperDistill introduces a scalar estimator $c_\gamma(t; \theta)$ to correct the scale:

$$c_\gamma(t; \theta) = \theta \cdot \|v_t\|_2 \cdot \frac{1 - \gamma^t}{1 - \gamma},$$

where $v_t = \nabla_{w_t} L_{\text{val}}(w_t, \lambda)$ and $\theta$ is a learnable parameter fitted periodically by minimizing the discrepancy between the estimated and true second-order terms on a short unrolled segment (Algorithm 4 in Lee et al., 2022).

### 2.5 Final Hypergradient

The complete online hypergradient estimate at step $t$ is:

$$\hat{g}_t = g_{\text{FO}} + c_\gamma(t; \theta) \cdot f_t(w_t^*, \mathcal{D}_t^*).$$

This requires only **one JVP per step**, uses **constant memory**, and supports **online updates** — $\lambda$ can be updated at every inner iteration without waiting for the inner loop to finish.

### 2.6 Weight Initialization via Reptile

HyperDistill also incorporates a Reptile-style (Nichol et al., 2018) meta-learning update for the initial weights $w_0$:

$$w_0 \leftarrow w_0 + \eta_{\text{reptile}} (w_T - w_0),$$

which learns an initialization that is amenable to fast adaptation across tasks.

---

## 3. Method 2: T1–T2 with Numerical DARTS Approximation

### 3.1 The T1–T2 Method

The T1–T2 method, introduced by Luketina et al. (2016), is a simple and efficient approach to gradient-based hyperparameter tuning. It approximates the hypergradient by considering only the influence of $\lambda$ on the *most recent* inner update step, ignoring the dependence of $w_{t-1}$ on $\lambda$:

$$\lambda_{t+1} = \lambda_t + \eta_2 \, (\nabla_\theta C_2) \, (\nabla_\lambda \nabla_\theta \tilde{C}_1),$$

where $C_2$ is the validation cost, $\tilde{C}_1$ is the training cost, and the gradient is taken through a single step of the inner optimization. This is equivalent to setting $T = 1$ in the full unrolled differentiation, or equivalently, approximating the inverse Hessian with the identity matrix.

Formally, the T1–T2 hypergradient is:

$$\hat{d}_\lambda L_{\text{val}} = \nabla_\lambda L_{\text{val}}(w_T, \lambda) + \nabla_{w_T} L_{\text{val}}(w_T, \lambda) \cdot B_T,$$

which retains only the last term ($t = T$) from the full response Jacobian sum. This makes it a special case of the Generalized Greedy method with $\gamma \to 0$ (see Section 4).

**Advantages:**
- Requires only a single JVP per outer step.
- Memory complexity $\mathcal{O}(P + H)$.
- No need for inner optimization convergence.

**Limitations:**
- Suffers from *short-horizon bias* (Wu et al., 2018): it only captures the local effect of $\lambda$ on the current step, ignoring long-range dependencies.

### 3.2 Relationship to Implicit Differentiation

Luketina et al. (2016) show that the T1–T2 update can be viewed as an approximation to the implicit differentiation approach of Bengio (2000), where the inverse Hessian $(\nabla^2_w L_{\text{train}})^{-1}$ is replaced by the identity matrix $I$. This connection is justified empirically by the observation that batch normalization tends to make the Hessian close to the identity.

### 3.3 Numerical DARTS Approximation

In the library's `T1T2Optimizer`, the second-order term $\nabla_{w_T} L_{\text{val}} \cdot B_T$ is computed using the **numerical finite-difference approximation** from DARTS (Liu et al., 2018). The DARTS paper proposes approximating the mixed second-order derivative via:

$$\nabla^2_{\lambda, w} L_{\text{train}}(w, \lambda) \cdot v \approx \frac{\nabla_\lambda L_{\text{train}}(w^+, \lambda) - \nabla_\lambda L_{\text{train}}(w^-, \lambda)}{2\epsilon},$$

where $w^\pm = w \pm \epsilon \, v$ and $v = \nabla_{w'} L_{\text{val}}(w', \lambda)$.

In the library implementation, this is adapted to compute $B_T \cdot v$ by perturbing $\lambda$ instead of $w$:

$$\frac{\partial \Phi}{\partial \lambda_i} \cdot v \approx \frac{\Phi(w, \lambda + \epsilon e_i) - \Phi(w, \lambda - \epsilon e_i)}{2\epsilon},$$

where $e_i$ is the $i$-th standard basis vector. This avoids computing the full Jacobian $B_T$ and reduces the cost to $\mathcal{O}(H)$ forward passes through $\Phi$, each of which is cheap compared to a full backward pass.

---

## 4. Method 3: Generalized Greedy Gradient-Based Optimization

### 4.1 Motivation

The Generalized Greedy method (Anonymous, under review at ICLR 2025) addresses the limitations of both truncated differentiation (short-horizon bias) and implicit differentiation (requires inner convergence). It generalizes the T1–T2 method by accumulating greedy gradient contributions from *every* step of the inner optimization trajectory.

### 4.2 Hypergradient Approximation

The method replaces the expensive Jacobian products $\prod_{k=t+1}^{T} A_k$ with a scalar decay $\gamma^{T-t}$ (similar to HyperDistill's Hessian identity approximation) and additionally replaces the terminal validation gradient $\nabla_{w_T} L_{\text{val}}$ with the local gradient $\nabla_{w_t} L_{\text{val}}$ at each step:

$$\hat{d}_\lambda L_{\text{val}}(w_T, \lambda; \gamma) = \nabla_\lambda L_{\text{val}}(w_T, \lambda) + \sum_{t=1}^{T} \gamma^{T-t} \, \nabla_{w_t} L_{\text{val}}(w_t, \lambda) \cdot B_t.$$

This is a weighted sum of *locally optimal greedy gradients*, where each term $\nabla_{w_t} L_{\text{val}}(w_t, \lambda) \cdot B_t$ captures the local effect of $\lambda$ on the validation loss at step $t$, and the exponential weighting $\gamma^{T-t}$ gives more importance to recent steps.

### 4.3 Generalization of T1–T2

The Generalized Greedy hypergradient strictly generalizes T1–T2. As shown in Proposition 4.1 of the paper:

$$\lim_{\gamma \to 0^+} \hat{d}_\lambda L_{\text{val}}(w_T, \lambda; \gamma) = \nabla_\lambda L_{\text{val}}(w_T, \lambda) + \nabla_{w_T} L_{\text{val}}(w_T, \lambda) \cdot B_T,$$

which is exactly the T1–T2 hypergradient. Thus, $\gamma$ interpolates between the single-step T1–T2 approximation ($\gamma \to 0$) and a full-trajectory accumulation ($\gamma \to 1$).

### 4.4 Sufficient Descent Guarantee

Under standard assumptions (smoothness, strong convexity of $L_{\text{val}}$, Lipschitz continuity of $B_t$, and the Hessian identity approximation $\nabla^2_w L_{\text{train}} = I$), the paper proves that the approximate hypergradient is a *sufficient descent direction*:

**Theorem 4.6.** Suppose $\gamma = 1 - \eta \in (0, 1)$. Then there exists a sufficiently large $T$ and a universal constant $c > 0$ such that:

$$d_\lambda L_{\text{val}}(w_T, \lambda) \cdot \hat{d}_\lambda L_{\text{val}}(w_T, \lambda; \gamma)^\top \geq c \, \|d_\lambda L_{\text{val}}(w_T, \lambda)\|_2^2.$$

This guarantees that updating $\lambda$ in the direction of $\hat{d}_\lambda$ will decrease the true validation loss, provided the inner optimization runs for sufficiently many steps.

### 4.5 Complexity

The method requires $T$ JVPs per outer step (one per inner step), which is more expensive than T1–T2 (1 JVP) but avoids the $\mathcal{O}(PT)$ memory of reverse-mode differentiation. The space complexity is $\mathcal{O}(P + H)$, matching T1–T2 and HyperDistill.

### 4.6 Experimental Findings

Experiments on toy problems, data hyper-cleaning (MNIST, Fashion-MNIST), and few-shot meta-learning (Omniglot) show that the Generalized Greedy method with $\gamma = 0.9$ consistently outperforms T1–T2, first-order baselines, and IFT-based methods, while maintaining comparable computational cost.

---

## 5. Comparison of Methods

| Aspect | HyperDistill | T1–T2 + DARTS | Generalized Greedy |
|---|---|---|---|
| **Paper** | Lee et al., ICLR 2022 | Luketina et al., ICML 2016 + Liu et al., ICLR 2019 | Anonymous, ICLR 2025 (under review) |
| **Core idea** | Distill $g_{\text{SO}}$ into single JVP at EMA weight point | One-step lookahead with finite-difference Jacobian | Accumulate weighted greedy gradients over trajectory |
| **Horizon** | Online (streaming) | Single step | Full trajectory ($T$ steps) |
| **JVPs / outer step** | 1 | 1 | $T$ |
| **Key approximation** | $A_k \approx \gamma I$ + distilled weight + scale estimator | $A_k$ ignored (only last step) + finite-difference $B_T$ | $\prod A_k \approx \gamma^{T-t} I$ + local $\nabla_w L_{\text{val}}$ |
| **Short-horizon bias** | No | Yes | No |
| **Online updates** | Yes | Yes (per outer step) | No (requires full inner rollout) |
| **Hyperparameters** | $\gamma$, $\theta$ (auto-fitted), Reptile $\eta$ | $\epsilon$ (finite-diff step) | $\gamma$ |
| **Library class** | `OnlineHypergradientOptimizer` | `T1T2Optimizer` | `GreedyOptimizer` |

---

## 6. Library Architecture and Design

### 6.1 Overview

GradHPO is organized as a Python package (`mylib`) with a modular architecture that separates core abstractions from algorithm implementations and utility functions:

```
src/mylib/
├── __init__.py          # Public API exports
├── train.py             # Training utilities
├── core/
│   ├── __init__.py
│   ├── base.py          # BilevelOptimizer ABC
│   ├── state.py         # BilevelState dataclass
│   └── types.py         # Type definitions
├── algorithms/
│   ├── __init__.py
│   ├── online.py        # OnlineHypergradientOptimizer (HyperDistill)
│   ├── t1t2.py          # T1T2Optimizer
│   ├── greedy.py        # GreedyOptimizer
│   └── baselines.py     # FOOptimizer, OneStepOptimizer
└── utils/
    ├── __init__.py
    └── gradients.py     # PyTree operations, VJP helpers, EMA updates
```

### 6.2 Core Abstractions

#### BilevelState

The `BilevelState` dataclass is the central state container, holding all quantities needed across optimization steps:

- **`params`** (`PyTree`): Current model parameters $w_t$.
- **`hyperparams`** (`PyTree`): Current hyperparameters $\lambda$.
- **`inner_step`** (`int`): Current inner iteration counter.
- **`outer_step`** (`int`): Current outer iteration counter.
- **`metrics`** (`dict`): Accumulated metrics (losses, gradient norms, etc.).

The state is immutable — each `step()` call returns a new `BilevelState` rather than mutating in place. This design is essential for compatibility with JAX's functional transformation model (`jax.jit`, `jax.grad`, `jax.vmap`).

#### BilevelOptimizer (Abstract Base Class)

All optimizers inherit from `BilevelOptimizer`, which defines a unified interface:

- **`init(params, hyperparams) → BilevelState`**: Initialize the optimization state.
- **`step(state, train_batch, val_batch, ...) → BilevelState`**: Perform one complete bilevel optimization step (inner update + hypergradient computation + outer update).
- **`compute_hypergradient(state, val_batch, ...) → PyTree`**: Compute the hypergradient estimate (method-specific).

This abstraction allows users to swap between optimization algorithms with minimal code changes.

#### Type Definitions

The library defines several core types for clarity and type safety:

- **`PyTree`**: JAX pytree (nested structure of arrays) — used for both parameters and hyperparameters.
- **`LossFn`**: Callable `(params, hyperparams, batch) → scalar` — the loss function signature.
- **`DataBatch`**: Named tuple with `inputs` and `targets` fields.
- **`LossFunctions`**: Named tuple bundling `train_loss` and `val_loss`.

### 6.3 Algorithm Implementations

Each optimizer class encapsulates its specific hypergradient computation logic while sharing the common `BilevelOptimizer` interface:

- **`OnlineHypergradientOptimizer`** (421 lines): Implements the full HyperDistill algorithm including distilled weight point EMA, linear scaling estimator with periodic $\theta$ fitting, and optional Reptile weight initialization.

- **`T1T2Optimizer`** (379 lines): Implements T1–T2 with numerical DARTS finite-difference approximation. The `_numerical_darts_approximation` method perturbs each hyperparameter element by $\pm\epsilon$ and computes the Jacobian-vector product via central differences.

- **`GreedyOptimizer`** (279 lines): Implements the Generalized Greedy method with configurable $\gamma$ decay. The `_rollout` method performs $T$ inner steps, and `_compute_hypergradient_from_rollout` accumulates the weighted greedy gradients.

- **`FOOptimizer`** and **`OneStepOptimizer`** (317 lines combined): Baseline implementations. `FOOptimizer` uses only the first-order term $g_{\text{FO}}$, while `OneStepOptimizer` implements the original T1–T2 without the DARTS approximation.

### 6.4 Utility Functions

The `utils/gradients.py` module provides essential building blocks:

- **`tree_l2_norm(tree)`**: Compute the L2 norm of a pytree.
- **`tree_normalize(tree)`**: Normalize a pytree to unit norm.
- **`tree_dot(a, b)`**: Inner product between two pytrees.
- **`tree_zeros_like(tree)`**: Create a zero-valued pytree with the same structure.
- **`tree_lerp(a, b, t)`**: Linear interpolation between pytrees.
- **`vjp_wrt_lambda(update_fn, w, lam, batch, v)`**: Vector-Jacobian product of the update function with respect to $\lambda$.
- **`vjp_wrt_both(update_fn, w, lam, batch, v_w, v_lam)`**: VJP with respect to both $w$ and $\lambda$.
- **`update_w_star(w_star, w_prev, gamma, step)`**: EMA update for the distilled weight point in HyperDistill.

### 6.5 Technology Stack

| Component | Choice | Rationale |
|---|---|---|
| **Autodiff engine** | JAX ≥ 0.4.0 | Functional transformations (`jit`, `grad`, `vmap`), XLA compilation, composable VJPs/JVPs |
| **Gradient transforms** | Optax ≥ 0.1.7 | Composable optimizer chains (Adam, SGD, etc.), learning rate schedules |
| **Testing utilities** | Chex ≥ 0.1.8 | Pytree assertions, dataclass validation, shape/dtype checking |
| **Numerical backend** | NumPy ≥ 1.24.0 | Array operations, data preprocessing |

The choice of **JAX** as the autodiff backend is central to the library's design. JAX's functional programming model — where transformations like `jax.grad`, `jax.jit`, and `jax.vmap` compose cleanly — aligns naturally with the bilevel optimization workflow. In particular:

- **`jax.grad`** and **`jax.vjp`** enable efficient computation of the VJPs needed for hypergradient estimation.
- **`jax.jit`** compiles the inner loop and hypergradient computation for GPU/TPU acceleration.
- The **pytree** abstraction allows parameters and hyperparameters to be arbitrary nested structures (dicts, lists, named tuples), which is essential for real neural network architectures.

**Optax** provides a clean API for composing gradient transformations (e.g., Adam with gradient clipping and learning rate decay), which is used for both inner and outer optimization loops.

---

## 7. Conclusion

GradHPO provides a unified, JAX-native implementation of three complementary approaches to gradient-based hyperparameter optimization. The methods span a spectrum of approximation strategies:

- **HyperDistill** offers the most sophisticated approximation with online updates and constant-cost hypergradient estimation, at the expense of additional hyperparameters ($\gamma$, $\theta$).
- **T1–T2 + DARTS** provides the simplest and cheapest approximation (one JVP, no trajectory storage), suitable when short-horizon bias is acceptable.
- **Generalized Greedy** strikes a middle ground, accumulating information from the full trajectory with a tunable decay parameter $\gamma$ and provable descent guarantees.

The library's modular design — with a shared `BilevelOptimizer` interface, immutable `BilevelState`, and JAX-compatible functional style — makes it straightforward to experiment with different methods and extend the framework with new algorithms.

---

## References

1. **Lee, H. B., Lee, H., Shin, J., Yang, E., Hospedales, T., & Hwang, S. J.** (2022). Online Hyperparameter Meta-Learning with Hypergradient Distillation. *International Conference on Learning Representations (ICLR)*.

2. **Luketina, J., Berglund, M., Greff, K., & Raiko, T.** (2016). Scalable Gradient-Based Tuning of Continuous Regularization Hyperparameters. *International Conference on Machine Learning (ICML)*, pp. 2952–2960.

3. **Liu, H., Simonyan, K., & Yang, Y.** (2019). DARTS: Differentiable Architecture Search. *International Conference on Learning Representations (ICLR)*.

4. **Anonymous.** (2025). Generalized Greedy Gradient-Based Hyperparameter Optimization. *Under review at ICLR 2025*.

5. **Franceschi, L., Donini, M., Frasconi, P., & Pontil, M.** (2017). Forward and Reverse Gradient-Based Hyperparameter Optimization. *International Conference on Machine Learning (ICML)*, pp. 1165–1173.

6. **Nichol, A., Achiam, J., & Schulman, J.** (2018). On First-Order Meta-Learning Algorithms. *arXiv preprint arXiv:1803.02999*.

7. **Bengio, Y.** (2000). Gradient-Based Optimization of Hyperparameters. *Neural Computation*, 12(8), 1889–1900.

8. **Wu, Y., Ren, M., Liao, R., & Grosse, R.** (2018). Understanding Short-Horizon Bias in Stochastic Meta-Optimization. *arXiv preprint arXiv:1803.02021*.

9. **Lorraine, J., Vicol, P., & Duvenaud, D.** (2020). Optimizing Millions of Hyperparameters by Implicit Differentiation. *International Conference on Artificial Intelligence and Statistics (AISTATS)*, pp. 1540–1552.

10. **Maclaurin, D., Duvenaud, D., & Adams, R.** (2015). Gradient-Based Hyperparameter Optimization Through Reversible Learning. *International Conference on Machine Learning (ICML)*, pp. 2113–2122.

11. **Shaban, A., Cheng, C.-A., Hatch, N., & Boots, B.** (2019). Truncated Back-Propagation for Bilevel Optimization. *International Conference on Artificial Intelligence and Statistics (AISTATS)*, pp. 1723–1732.
