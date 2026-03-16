#!/usr/bin/env python3
"""POC demonstration of HyperDistill for per-parameter learning rate optimisation.

Task
----
Train a small MLP on synthetic multiclass classification data.
The *hyperparameter* is a per-parameter learning rate vector (same shape as
the model weights).  The inner loop performs SGD with these learned rates,
and the outer loop uses HyperDistill / baselines to tune them online.

We compare three methods:
  1. Fixed LR  — no hyperparameter optimisation (baseline).
  2. 1-step    — one-step lookahead (Luketina et al., 2016).
  3. HyperDistill — our algorithm (Lee et al., ICLR 2022).

Usage
-----
    python code/demo_hyperdistill.py          # runs the demo
    python code/demo_hyperdistill.py --plot    # runs + saves convergence plot
"""

from __future__ import annotations

import argparse
import sys
import os

import jax
import jax.numpy as jnp
import numpy as np

# ── make the library importable when running from the repo root ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from mylib.hyperdistill import (
    run_hyperdistill,
    run_baseline,
    tree_zeros_like,
)

# ---------------------------------------------------------------------------
# Simple MLP model (pure JAX, no Flax/Haiku)
# ---------------------------------------------------------------------------

def init_mlp(key, in_dim: int, hidden_dim: int, out_dim: int) -> dict:
    """Xavier-initialised two-layer MLP: Linear -> ReLU -> Linear."""
    k1, k2 = jax.random.split(key)
    params = {
        'w1': jax.random.normal(k1, (in_dim, hidden_dim)) * jnp.sqrt(2.0 / in_dim),
        'b1': jnp.zeros(hidden_dim),
        'w2': jax.random.normal(k2, (hidden_dim, out_dim)) * jnp.sqrt(2.0 / hidden_dim),
        'b2': jnp.zeros(out_dim),
    }
    return params


def mlp_forward(params: dict, x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass returning logits."""
    h = jax.nn.relu(x @ params['w1'] + params['b1'])
    return h @ params['w2'] + params['b2']


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def cross_entropy_loss(params: dict, batch: tuple) -> jnp.ndarray:
    """Softmax cross-entropy loss (scalar)."""
    x, y = batch
    logits = mlp_forward(params, x)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(log_probs * y, axis=-1))


# ---------------------------------------------------------------------------
# Update function: SGD with per-parameter learning rates
# ---------------------------------------------------------------------------

def make_update_fn(loss_fn):
    """Return Phi(w, lr_params, batch) that does one SGD step
    with per-parameter LR = softplus(lr_params)."""

    def update_fn(w, lr_params, batch):
        grads = jax.grad(loss_fn)(w, batch)
        return jax.tree.map(
            lambda w_i, lr_i, g_i: w_i - jax.nn.softplus(lr_i) * g_i,
            w, lr_params, grads,
        )

    return update_fn


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_centres(key, n_features: int, n_classes: int):
    """Generate fixed class centres (shared between train and val)."""
    return jax.random.normal(key, (n_classes, n_features)) * 2.0


def make_data(key, centres: jnp.ndarray, n_samples: int, n_classes: int,
              noise_rate: float = 0.0):
    """Synthetic multiclass data with optional label noise.

    Returns (X, Y_onehot) where Y_onehot is one-hot encoded.
    """
    k1, k2, k3 = jax.random.split(key, 3)

    labels = jax.random.randint(k1, (n_samples,), 0, n_classes)
    X = centres[labels] + jax.random.normal(k2, (n_samples, centres.shape[1])) * 0.5

    # Optional label noise
    if noise_rate > 0:
        flip_mask = jax.random.bernoulli(k3, noise_rate, (n_samples,))
        random_labels = jax.random.randint(k3, (n_samples,), 0, n_classes)
        labels = jnp.where(flip_mask, random_labels, labels)

    Y = jax.nn.one_hot(labels, n_classes)
    return np.array(X), np.array(Y)


class BatchIterator:
    """Cycles through a dataset yielding random mini-batches."""

    def __init__(self, X, Y, batch_size: int, key):
        self.X, self.Y = jnp.array(X), jnp.array(Y)
        self.batch_size = batch_size
        self.key = key
        self.n = X.shape[0]

    def __call__(self):
        self.key, subkey = jax.random.split(self.key)
        idx = jax.random.randint(subkey, (self.batch_size,), 0, self.n)
        return (self.X[idx], self.Y[idx])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(params, X, Y):
    """Return (loss, accuracy) on the full dataset."""
    logits = mlp_forward(params, jnp.array(X))
    y_oh = jnp.array(Y)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.mean(jnp.sum(log_probs * y_oh, axis=-1))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y_oh, axis=-1))
    return float(loss), float(acc)


# ---------------------------------------------------------------------------
# Run one training method for M episodes, return val-loss curve
# ---------------------------------------------------------------------------

def run_fixed_lr(w_init, lam, T, M, lr_reptile,
                 update_fn, get_train, get_val, X_val, Y_val):
    """Baseline: fixed learning rate, no HPO."""
    phi = w_init
    losses = []
    for m in range(1, M + 1):
        w = phi
        for _ in range(T):
            w = update_fn(w, lam, get_train())
        phi = jax.tree.map(lambda p, wt: p - lr_reptile * (p - wt), phi, w)
        loss, _ = evaluate(w, X_val, Y_val)
        losses.append(loss)
    return losses


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='HyperDistill POC demo')
    parser.add_argument('--plot', action='store_true',
                        help='Save convergence plot to figures/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--M', type=int, default=60,
                        help='Number of outer episodes')
    parser.add_argument('--T', type=int, default=20,
                        help='Inner steps per episode')
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)

    # ── Hyperparameters ──
    N_FEATURES = 10
    N_CLASSES  = 5
    HIDDEN     = 32
    BATCH_SIZE = 128
    GAMMA      = 0.99
    LR_HYPER   = 3e-3
    LR_REPTILE = 1.0
    INIT_LR    = 0.05          # initial inner learning rate
    T, M       = args.T, args.M

    # ── Data (shared class centres for train and val) ──
    k_centres, k_data, k_model, k_batch = jax.random.split(key, 4)
    centres = make_centres(k_centres, N_FEATURES, N_CLASSES)
    k_train, k_val_data = jax.random.split(k_data)
    X_train, Y_train = make_data(k_train, centres, 500, N_CLASSES,
                                 noise_rate=0.2)
    X_val, Y_val = make_data(k_val_data, centres, 200, N_CLASSES,
                             noise_rate=0.0)

    # Each method gets its own batch iterators with the same seed
    def make_iterators(seed_offset=0):
        k = jax.random.PRNGKey(args.seed + seed_offset)
        k1, k2 = jax.random.split(k)
        gt = BatchIterator(X_train, Y_train, BATCH_SIZE, k1)
        gv = BatchIterator(X_val, Y_val, BATCH_SIZE, k2)
        return gt, gv

    # ── Model ──
    w_init = init_mlp(k_model, N_FEATURES, HIDDEN, N_CLASSES)

    # Per-parameter LR initialised so that softplus(lr) ≈ INIT_LR
    init_val = float(jnp.log(jnp.exp(INIT_LR) - 1.0))
    lam_init = jax.tree.map(lambda p: jnp.full_like(p, init_val), w_init)

    update_fn = make_update_fn(cross_entropy_loss)

    n_params = sum(p.size for p in jax.tree.leaves(w_init))
    print(f'Model params:  {n_params}')
    print(f'Hyper params:  {n_params}  (per-parameter LR)')
    print(f'Episodes M={M}, inner steps T={T}, gamma={GAMMA}')
    print()

    # ── 1. Fixed LR ──
    print('Running Fixed-LR baseline ...')
    gt, gv = make_iterators(100)
    fixed_losses = run_fixed_lr(
        w_init, lam_init, T, M, LR_REPTILE,
        update_fn, gt, gv, X_val, Y_val)

    # ── 2. One-step ──
    print('Running 1-step baseline ...')
    onestep_losses = []

    def onestep_cb(ep, w, lam, metrics):
        loss, _ = evaluate(w, X_val, Y_val)
        onestep_losses.append(loss)

    gt, gv = make_iterators(200)
    run_baseline(
        w_init, lam_init, T, M, LR_HYPER, LR_REPTILE,
        update_fn, cross_entropy_loss,
        gt, gv,
        method='one_step', callback=onestep_cb)

    # ── 3. HyperDistill ──
    print('Running HyperDistill ...')
    hd_losses = []

    def hd_cb(ep, w, lam, metrics):
        loss, _ = evaluate(w, X_val, Y_val)
        hd_losses.append(loss)

    gt, gv = make_iterators(300)
    run_hyperdistill(
        w_init, lam_init, GAMMA, T, M, LR_HYPER, LR_REPTILE,
        update_fn, cross_entropy_loss,
        gt, gv,
        estimation_period=10, callback=hd_cb)

    # ── Final evaluation on full val set ──
    print()
    print('Final validation (full dataset):')
    for name, losses in [('Fixed LR', fixed_losses),
                         ('1-step', onestep_losses),
                         ('HyperDistill', hd_losses)]:
        mean_last5 = np.mean(losses[-5:])
        best = min(losses)
        print(f'  {name:14s}  last-5 avg loss={mean_last5:.4f}  '
              f'best loss={best:.4f}')

    # ── Plot ──
    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('matplotlib not installed — skipping plot.')
            return

        def smooth(vals, window=5):
            """Simple moving average for smoother curves."""
            kernel = np.ones(window) / window
            return np.convolve(vals, kernel, mode='valid')

        fig, ax = plt.subplots(figsize=(8, 5))
        w = 5  # smoothing window
        eps_smooth = range(w, M + 1)
        ax.plot(eps_smooth, smooth(fixed_losses, w),
                label='Fixed LR', linewidth=2)
        ax.plot(eps_smooth, smooth(onestep_losses, w),
                label='1-step', linewidth=2)
        ax.plot(eps_smooth, smooth(hd_losses, w),
                label='HyperDistill', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Validation loss (smoothed)')
        ax.set_title('HyperDistill POC — per-parameter LR optimisation')
        ax.legend()
        ax.grid(alpha=0.3)

        os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'figures'),
                    exist_ok=True)
        path = os.path.join(os.path.dirname(__file__), '..',
                            'figures', 'hyperdistill_poc.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f'Plot saved to {path}')


if __name__ == '__main__':
    main()
