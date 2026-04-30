"""Tests for OnlineHypergradientOptimizer (HyperDistill) covering the
optax-driven inner/outer paths and the full ``run`` training loop.
"""

import jax.numpy as jnp
import optax

from gradhpo.algorithms.online import OnlineHypergradientOptimizer


class TestOnlineWithOptax:
    def test_step_uses_optax_inner_and_outer(
        self, model, hyperparams, batch, loss_fns_with_l2,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = OnlineHypergradientOptimizer(
            inner_optimizer=optax.sgd(0.05),
            outer_optimizer=optax.sgd(1e-2),
            gamma=0.9, T=3,
        )
        state = opt.init(model, hyperparams)
        new_state = opt.step(state, batch, batch, train_loss, val_loss)

        assert state.inner_opt_state is not None
        assert state.outer_opt_state is not None
        # Inner state was updated.
        assert new_state.inner_opt_state is not state.inner_opt_state
        # Outer state was updated.
        assert new_state.outer_opt_state is not state.outer_opt_state
        assert new_state.step == 1
        # Hyperparams moved because val loss depends on them.
        assert not jnp.allclose(
            new_state.hyperparams['w1'], hyperparams['w1'],
        )


class TestOnlineRun:
    def test_run_short(
        self, model, hyperparams, loss_fns, update_fn, batch_iter,
    ):
        train_loss, val_loss = loss_fns
        opt = OnlineHypergradientOptimizer(
            update_fn=update_fn, gamma=0.95,
            estimation_period=1, T=3,
        )
        state = opt.init(model, hyperparams)

        callbacks = []
        final = opt.run(
            state, M=2,
            get_train_batch=batch_iter,
            get_val_batch=batch_iter,
            train_loss_fn=train_loss,
            val_loss_fn=val_loss,
            lr_reptile=1.0, lr_hyper=1e-3,
            callback=lambda m, st: callbacks.append(m),
        )

        assert callbacks == [1, 2]
        # Theta was estimated and stored.
        assert jnp.isfinite(final.get_metric('theta'))
        # Final state has phi memorised.
        assert final.get_metric('phi') is not None
