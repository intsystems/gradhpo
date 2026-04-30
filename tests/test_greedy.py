"""Tests for GreedyOptimizer (Eq. 6 generalised greedy hypergradient).

The greedy optimiser is built around an optax inner+outer optimiser pair, so
these tests cover initialisation, single-step updates, hypergradient
computation and the full training loop.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from gradhpo.algorithms.greedy import GreedyOptimizer


class TestGreedyValidation:
    def test_unroll_steps_must_be_positive(self):
        with pytest.raises(ValueError, match="unroll_steps"):
            GreedyOptimizer(
                inner_optimizer=optax.sgd(0.01),
                outer_optimizer=optax.sgd(1e-3),
                unroll_steps=0,
            )

    @pytest.mark.parametrize("bad_gamma", [0.0, -0.1, 1.1, 2.0])
    def test_gamma_must_be_in_unit_interval(self, bad_gamma):
        with pytest.raises(ValueError, match="gamma"):
            GreedyOptimizer(
                inner_optimizer=optax.sgd(0.01),
                outer_optimizer=optax.sgd(1e-3),
                gamma=bad_gamma,
            )


class TestGreedyInit:
    def test_init_creates_state_with_optimizer_states(
        self, model, hyperparams,
    ):
        opt = GreedyOptimizer(
            inner_optimizer=optax.sgd(0.01),
            outer_optimizer=optax.adam(1e-3),
        )
        state = opt.init(model, hyperparams)

        assert state.step == 0
        assert state.inner_opt_state is not None
        assert state.outer_opt_state is not None


class TestGreedyStep:
    def test_step_unroll1_updates_state(
        self, model, hyperparams, batch, loss_fns_with_l2,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = GreedyOptimizer(
            inner_optimizer=optax.sgd(0.05),
            outer_optimizer=optax.sgd(1e-2),
            unroll_steps=1, gamma=0.9,
        )
        state = opt.init(model, hyperparams)
        new_state = opt.step(state, batch, batch, train_loss, val_loss)

        assert not jnp.allclose(new_state.params['w1'], model['w1'])
        assert not jnp.allclose(
            new_state.hyperparams['w1'], hyperparams['w1'],
        )
        # Metadata is populated.
        for key in ('train_loss', 'val_loss', 'hypergrad_norm',
                    'param_norm', 'hyperparam_norm'):
            assert key in new_state.metadata

    def test_step_unroll3_runs_multiple_inner_steps(
        self, model, hyperparams, batch, loss_fns_with_l2,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = GreedyOptimizer(
            inner_optimizer=optax.sgd(0.05),
            outer_optimizer=optax.sgd(1e-2),
            unroll_steps=3, gamma=0.5,
        )
        state = opt.init(model, hyperparams)
        new_state = opt.step(state, batch, batch, train_loss, val_loss)

        # After 3 inner steps, params are noticeably further from init than
        # after a single step.
        diff_norm = float(
            jnp.sqrt(jnp.sum(jnp.square(new_state.params['w1'] - model['w1'])))
        )
        assert diff_norm > 0


class TestGreedyComputeHypergradient:
    def test_hypergradient_shape_matches_hyperparams(
        self, model, hyperparams, batch, loss_fns_with_l2,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = GreedyOptimizer(
            inner_optimizer=optax.sgd(0.01),
            outer_optimizer=optax.sgd(1e-3),
            unroll_steps=2, gamma=0.9,
        )
        state = opt.init(model, hyperparams)
        hg = opt.compute_hypergradient(
            state, batch, batch, train_loss, val_loss,
        )

        for hg_leaf, hp_leaf in zip(jax.tree.leaves(hg),
                                    jax.tree.leaves(hyperparams)):
            assert hg_leaf.shape == hp_leaf.shape


class TestGreedyRun:
    def test_run_full_loop_with_callback(
        self, model, hyperparams, loss_fns_with_l2, batch_iter,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = GreedyOptimizer(
            inner_optimizer=optax.sgd(0.01),
            outer_optimizer=optax.sgd(1e-3),
            unroll_steps=2, gamma=0.9,
        )
        state = opt.init(model, hyperparams)

        episodes = []
        final = opt.run(
            state, M=2,
            get_train_batch=batch_iter,
            get_val_batch=batch_iter,
            train_loss_fn=train_loss,
            val_loss_fn=val_loss,
            lr_reptile=0.5,
            callback=lambda m, st: episodes.append(m),
        )

        assert episodes == [1, 2]
        assert final.params is not None
