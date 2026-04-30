"""Tests for the T1T2 optimizer's optax inner/outer paths and run loop."""

import optax

from gradhpo.algorithms.t1t2 import T1T2Optimizer


class TestT1T2WithOptax:
    def test_step_with_optax(
        self, model, hyperparams, batch, loss_fns_with_l2,
    ):
        train_loss, val_loss = loss_fns_with_l2
        opt = T1T2Optimizer(
            inner_optimizer=optax.sgd(0.05),
            outer_optimizer=optax.sgd(1e-2),
            gamma=0.9, T=3, eps=1e-3,
        )
        state = opt.init(model, hyperparams)
        new_state = opt.step(state, batch, batch, train_loss, val_loss)

        assert new_state.step == 1
        assert new_state.inner_opt_state is not state.inner_opt_state
        assert new_state.outer_opt_state is not state.outer_opt_state


class TestT1T2Run:
    def test_run_executes_callback_each_episode(
        self, model, hyperparams, loss_fns, update_fn, batch_iter,
    ):
        train_loss, val_loss = loss_fns
        opt = T1T2Optimizer(update_fn=update_fn, gamma=0.9, T=2, eps=1e-3)
        state = opt.init(model, hyperparams)

        recorded = []
        opt.run(
            state, M=3,
            get_train_batch=batch_iter,
            get_val_batch=batch_iter,
            train_loss_fn=train_loss,
            val_loss_fn=val_loss,
            lr_reptile=1.0, lr_hyper=1e-3,
            callback=lambda m, st: recorded.append(m),
        )

        assert recorded == [1, 2, 3]
