"""Tests for the FOOptimizer and OneStepOptimizer baselines.

Cover the optax-driven inner/outer code paths and the full ``run`` training
loop, which the OOP-API tests in ``test_hyperdistill.py`` do not exercise.
"""

import jax
import jax.numpy as jnp
import optax

from gradhpo.algorithms.baselines import FOOptimizer, OneStepOptimizer


# ---------------------------------------------------------------------------
# FOOptimizer
# ---------------------------------------------------------------------------


class TestFOOptimizerWithOptax:
    def test_init_with_optax_optimizers(self, model, hyperparams):
        inner = optax.sgd(0.01)
        outer = optax.adam(1e-3)
        opt = FOOptimizer(inner_optimizer=inner, outer_optimizer=outer)

        state = opt.init(model, hyperparams)

        assert state.inner_opt_state is not None
        assert state.outer_opt_state is not None
        assert state.step == 0

    def test_step_with_optax_outer_changes_params(
        self, model, hyperparams, batch, loss_fns_with_l2,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = FOOptimizer(
            inner_optimizer=optax.sgd(0.05),
            outer_optimizer=optax.sgd(1e-2),
        )
        state = opt.init(model, hyperparams)
        new_state = opt.step(state, batch, batch, train_loss, val_loss)

        # Inner step changed weights.
        assert not jnp.allclose(new_state.params['w1'], model['w1'])
        # Outer optimiser updated hyperparams (val loss depends on them).
        assert not jnp.allclose(
            new_state.hyperparams['w1'], hyperparams['w1'],
        )
        assert new_state.step == 1

    def test_compute_hypergradient_with_optax(
        self, model, hyperparams, batch, loss_fns_with_l2,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = FOOptimizer(inner_optimizer=optax.sgd(0.01))
        state = opt.init(model, hyperparams)

        hg = opt.compute_hypergradient(
            state, batch, batch, train_loss, val_loss,
        )

        assert any(
            float(jnp.sum(jnp.abs(leaf))) > 0
            for leaf in jax.tree.leaves(hg)
        )

    def test_run_loop_executes(
        self, model, hyperparams, loss_fns, update_fn, batch_iter,
    ):
        train_loss, val_loss = loss_fns
        opt = FOOptimizer(update_fn=update_fn)
        state = opt.init(model, hyperparams)

        callbacks = []

        def cb(m, st):
            callbacks.append((m, float(st.get_metric('val_loss'))))

        final = opt.run(
            state, M=2, T=2,
            get_train_batch=batch_iter, get_val_batch=batch_iter,
            train_loss_fn=train_loss, val_loss_fn=val_loss,
            lr_reptile=0.5, lr_hyper=1e-3, callback=cb,
        )

        assert len(callbacks) == 2
        # Reptile mixed phi back to params.
        assert final.params is not None


# ---------------------------------------------------------------------------
# OneStepOptimizer
# ---------------------------------------------------------------------------


class TestOneStepOptimizerWithOptax:
    def test_init_with_optax_optimizers(self, model, hyperparams):
        inner = optax.sgd(0.01)
        outer = optax.adam(1e-3)
        opt = OneStepOptimizer(inner_optimizer=inner, outer_optimizer=outer)
        state = opt.init(model, hyperparams)

        assert state.inner_opt_state is not None
        assert state.outer_opt_state is not None

    def test_step_with_optax_outer(
        self, model, hyperparams, batch, loss_fns_with_l2,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = OneStepOptimizer(
            inner_optimizer=optax.sgd(0.05),
            outer_optimizer=optax.sgd(1e-2),
        )
        state = opt.init(model, hyperparams)
        new_state = opt.step(state, batch, batch, train_loss, val_loss)

        assert not jnp.allclose(new_state.params['w1'], model['w1'])
        # Hyperparams change because val loss depends on them.
        assert not jnp.allclose(
            new_state.hyperparams['w1'], hyperparams['w1'],
        )

    def test_compute_hypergradient_with_optax(
        self, model, hyperparams, batch, loss_fns_with_l2,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = OneStepOptimizer(inner_optimizer=optax.sgd(0.01))
        state = opt.init(model, hyperparams)

        hg = opt.compute_hypergradient(
            state, batch, batch, train_loss, val_loss,
        )

        assert all(leaf.shape == p.shape
                   for leaf, p in zip(jax.tree.leaves(hg),
                                      jax.tree.leaves(hyperparams)))

    def test_run_loop_executes(
        self, model, hyperparams, loss_fns, update_fn, batch_iter,
    ):
        train_loss, val_loss = loss_fns
        opt = OneStepOptimizer(update_fn=update_fn)
        state = opt.init(model, hyperparams)

        cbs = []
        final = opt.run(
            state, M=2, T=2,
            get_train_batch=batch_iter, get_val_batch=batch_iter,
            train_loss_fn=train_loss, val_loss_fn=val_loss,
            lr_reptile=0.5, lr_hyper=1e-3,
            callback=lambda m, st: cbs.append(m),
        )

        assert cbs == [1, 2]
        assert final.params is not None
