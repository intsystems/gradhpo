"""gradhpo: gradient-based hyperparameter optimization in JAX.

Public API: import the algorithm classes, ``BilevelState`` and pytree
helpers directly from the top-level package, e.g. ``from gradhpo import
OnlineHypergradientOptimizer, T1T2Optimizer, GreedyOptimizer``.
"""

__version__ = '0.1.3'

from gradhpo.algorithms import (
    FOOptimizer,
    GreedyOptimizer,
    OnlineHypergradientOptimizer,
    OneStepOptimizer,
    T1T2Optimizer,
)
from gradhpo.core import (
    BilevelOptimizer,
    BilevelState,
    DataBatch,
    LossFn,
    LossFunctions,
    MetricDict,
    PyTree,
)
from gradhpo.utils import (
    tree_dot,
    tree_l2_norm,
    tree_lerp,
    tree_normalize,
    tree_zeros_like,
    update_w_star,
    vjp_wrt_both,
    vjp_wrt_lambda,
)

__all__ = [
    '__version__',
    # Algorithms
    'OnlineHypergradientOptimizer',
    'T1T2Optimizer',
    'GreedyOptimizer',
    'FOOptimizer',
    'OneStepOptimizer',
    # Core
    'BilevelOptimizer',
    'BilevelState',
    'DataBatch',
    'LossFn',
    'LossFunctions',
    'MetricDict',
    'PyTree',
    # Utils
    'tree_dot',
    'tree_l2_norm',
    'tree_lerp',
    'tree_normalize',
    'tree_zeros_like',
    'update_w_star',
    'vjp_wrt_both',
    'vjp_wrt_lambda',
]
