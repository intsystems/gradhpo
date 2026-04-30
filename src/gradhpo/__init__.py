__version__ = '0.1.0'

# New OOP API
from gradhpo.core import (
    BilevelState,
    BilevelOptimizer,
    PyTree,
    LossFn,
    MetricDict,
    DataBatch,
    LossFunctions,
)
from gradhpo.algorithms import (
    OnlineHypergradientOptimizer,
    FOOptimizer,
    OneStepOptimizer,
)
from gradhpo.utils import (
    tree_l2_norm,
    tree_normalize,
    tree_dot,
    tree_zeros_like,
    tree_lerp,
    vjp_wrt_lambda,
    vjp_wrt_both,
    update_w_star,
)
