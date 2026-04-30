"""Tests for ``BilevelState`` covering pytree registration and the public API.

These ensure the dataclass roundtrips through ``jax.tree`` operations cleanly,
which is required for ``jax.jit``/``jax.grad`` compatibility.
"""

import jax
import jax.numpy as jnp

from gradhpo.core.state import BilevelState


class TestBilevelStateAPI:
    def test_create_initialises_step_and_metadata(self):
        params = {'w': jnp.ones(3)}
        hp = {'lr': jnp.ones(3) * 0.01}
        state = BilevelState.create(params, hp, None, None)

        assert state.step == 0
        assert state.metadata == {}
        assert jnp.allclose(state.params['w'], 1.0)

    def test_update_keeps_unspecified_fields(self):
        state = BilevelState.create(
            {'w': jnp.ones(3)}, {'lr': jnp.ones(3)}, None, None,
        )
        new_state = state.update(step=42)
        assert jnp.allclose(new_state.params['w'], state.params['w'])
        assert new_state.step == 42

    def test_update_merges_metadata(self):
        state = BilevelState(
            params={}, hyperparams={},
            inner_opt_state=None, outer_opt_state=None,
            step=0, metadata={'a': 1, 'b': 2},
        )
        merged = state.update(metadata={'b': 99, 'c': 3})

        assert merged.metadata == {'a': 1, 'b': 99, 'c': 3}
        # Original state unchanged.
        assert state.metadata == {'a': 1, 'b': 2}

    def test_update_replaces_optimizer_states(self):
        state = BilevelState.create(
            {'w': jnp.ones(2)}, {'lr': jnp.ones(2)}, None, None,
        )
        new = state.update(
            inner_opt_state='inner-x',
            outer_opt_state='outer-y',
            hyperparams={'lr': jnp.zeros(2)},
        )
        assert new.inner_opt_state == 'inner-x'
        assert new.outer_opt_state == 'outer-y'
        assert jnp.allclose(new.hyperparams['lr'], 0.0)

    def test_get_metric_returns_default_when_missing(self):
        state = BilevelState.create({'w': jnp.zeros(1)}, {}, None, None)
        assert state.get_metric('not-there', default=7) == 7
        assert state.get_metric('not-there') is None


class TestBilevelStatePytree:
    def test_roundtrip_through_tree_flatten(self):
        state = BilevelState(
            params={'w': jnp.ones(2)},
            hyperparams={'lr': jnp.full((2,), 0.1)},
            inner_opt_state=None,
            outer_opt_state=None,
            step=3,
            metadata={'val_loss': jnp.array(0.5), 'theta': 1.5},
        )
        leaves, treedef = jax.tree_util.tree_flatten(state)
        rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

        assert rebuilt.step == state.step
        assert set(rebuilt.metadata.keys()) == set(state.metadata.keys())
        assert jnp.allclose(rebuilt.params['w'], state.params['w'])
        assert jnp.isclose(rebuilt.metadata['val_loss'],
                           state.metadata['val_loss'])

    def test_jax_tree_map_doubles_arrays(self):
        state = BilevelState(
            params={'w': jnp.ones(2)},
            hyperparams={'lr': jnp.full((2,), 0.1)},
            inner_opt_state=None,
            outer_opt_state=None,
            step=1,
            metadata={'loss': jnp.array(2.0)},
        )
        doubled = jax.tree.map(lambda x: x * 2.0, state)

        assert jnp.allclose(doubled.params['w'], 2.0)
        assert jnp.allclose(doubled.hyperparams['lr'], 0.2)
        assert jnp.isclose(doubled.metadata['loss'], 4.0)
        # Step is aux data and should be preserved as-is.
        assert doubled.step == 1

    def test_works_inside_jit(self):
        @jax.jit
        def add_one_to_step(state):
            return state.update(step=state.step + 1)

        state = BilevelState.create(
            {'w': jnp.ones(2)}, {'lr': jnp.ones(2)}, None, None,
        )
        new_state = add_one_to_step(state)
        assert new_state.step == 1
