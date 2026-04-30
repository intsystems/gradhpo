"""Tests that exercise the top-level ``import gradhpo`` public surface."""

import gradhpo


def test_version_string():
    assert isinstance(gradhpo.__version__, str)
    assert gradhpo.__version__.count('.') >= 1


def test_top_level_reexports_exist():
    expected = [
        'BilevelState',
        'BilevelOptimizer',
        'OnlineHypergradientOptimizer',
        'FOOptimizer',
        'OneStepOptimizer',
        'tree_l2_norm',
        'tree_normalize',
        'tree_dot',
        'tree_zeros_like',
        'tree_lerp',
        'vjp_wrt_lambda',
        'vjp_wrt_both',
        'update_w_star',
        'LossFunctions',
    ]
    for name in expected:
        assert hasattr(gradhpo, name), f'gradhpo.{name} missing'


def test_loss_functions_namedtuple():
    pair = gradhpo.LossFunctions(train_loss=lambda *a: 0.0,
                                 val_loss=lambda *a: 0.0)
    assert callable(pair.train_loss)
    assert callable(pair.val_loss)


def test_bilevel_optimizer_is_abstract():
    """``BilevelOptimizer`` should not be instantiable directly."""
    import pytest
    with pytest.raises(TypeError):
        gradhpo.BilevelOptimizer()
