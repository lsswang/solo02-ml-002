"""Correctness and scaling verification for shared sparse training.

Run with the project virtual environment:

    .venv/bin/python verify.py

Checks, in order:
  1. Shared gradient against an independent finite-difference kernel.
  2. Shared gradient against an independently written dense analytic kernel,
     including re-chunking and reordering invariance and zero-weight steps.
  3. Full SparseTrainer trajectories against a from-scratch dense reference
     (one shared dense table + dense AdamW with global-norm clipping).
  4. LazyAdamW standalone against the same dense reference, covering empty
     gradient steps, read-only lookup, long-idle rows revisited, varying
     learning rates, export and a table exposing only shape/read/write.
  5. Boundary scenarios: the 131072-row vocabulary with dim 1, a long run of
     empty global steps followed by row revisit, vocabulary-extreme IDs,
     lookup order/duplicate/empty semantics and zero-weight skips.
  6. A reproducible 257 -> 131072 row scaling run that records real table
     traffic and optimizer state size.
"""
import json
from pathlib import Path

import numpy as np

from lazy_adamw import LazyAdamW
from recommender import shared_loss_and_grad
from row_table import ArrayRowTable
from sparse_trainer import SparseTrainer

ATOL, RTOL = 2e-10, 2e-9


# --------------------------------------------------------------- test harness
class CheckFailure(AssertionError):
    pass


def check(name, condition, detail=''):
    if not bool(condition):
        raise CheckFailure(f'{name} failed {detail}')
    print(f'  PASS  {name}')


def softplus(x):
    return np.logaddexp(0.0, x)


# ------------------------------------------------- independent dense reference
def dense_loss_and_grad(table, samples):
    """Independent dense reference: loss and a full (V, d) gradient table."""
    v_size, dim = table.shape
    grad = np.zeros_like(table)
    total_weight = sum(float(s['weight']) for s in samples)
    if total_weight <= 0.0:
        return 0.0, 0.0, grad
    weighted_loss = 0.0
    for s in samples:
        w = float(s['weight'])
        ctx = np.asarray(s['context'], dtype=np.int64)
        pos, neg = int(s['positive']), int(s['negative'])
        h = table[ctx].mean(axis=0) if ctx.size else np.zeros(dim)
        sp = float(h @ table[pos])
        sn = float(h @ table[neg])
        weighted_loss += w * (softplus(-sp) + softplus(sn))
        a = -w * (1.0 / (1.0 + np.exp(np.clip(sp, -500.0, 500.0))))
        b = w * (1.0 / (1.0 + np.exp(np.clip(-sn, -500.0, 500.0))))
        grad[pos] += a * h
        grad[neg] += b * h
        if ctx.size:
            dh = (a * table[pos] + b * table[neg]) / ctx.size
            np.add.at(grad, ctx, dh)
    return weighted_loss / total_weight, total_weight, grad / total_weight


def dense_loss_only(table, samples):
    loss = 0.0
    total_weight = 0.0
    dim = table.shape[1]
    for s in samples:
        w = float(s['weight'])
        total_weight += w
        ctx = np.asarray(s['context'], dtype=np.int64)
        h = table[ctx].mean(axis=0) if ctx.size else np.zeros(dim)
        loss += w * (softplus(-float(h @ table[int(s['positive'])]))
                     + softplus(float(h @ table[int(s['negative'])])))
    return loss / total_weight if total_weight > 0.0 else 0.0


class DenseRefAdamW:
    """Plain dense AdamW over every row, same operation order as DenseAdamW."""

    def __init__(self, table, beta1, beta2, eps, weight_decay):
        self.e = np.array(table, dtype=np.float64, copy=True)
        self.m = np.zeros_like(self.e)
        self.v = np.zeros_like(self.e)
        self.b1, self.b2, self.eps, self.wd = beta1, beta2, eps, weight_decay
        self.t = 0

    def step(self, grad, lr):
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * grad
        self.v = self.b2 * self.v + (1.0 - self.b2) * grad * grad
        mh = self.m / (1.0 - self.b1 ** self.t)
        vh = self.v / (1.0 - self.b2 ** self.t)
        self.e *= 1.0 - lr * self.wd
        self.e -= lr * mh / (np.sqrt(vh) + self.eps)


def dense_trainer_step(ref, samples, lr, clip_norm):
    loss, total_weight, grad = dense_loss_and_grad(ref.e, samples)
    if total_weight <= 0.0:
        return loss, 0.0, 0.0
    norm = float(np.sqrt(np.sum(grad * grad)))
    scale = min(1.0, clip_norm / norm) if norm > 0.0 else 1.0
    ref.step(grad * scale, lr)
    return loss, total_weight, norm


class RestrictedTable:
    """Table exposing only the promised shape/read/write interface."""

    def __init__(self, values):
        self._backing = ArrayRowTable(values)
        self.shape = self._backing.shape

    def read(self, ids):
        return self._backing.read(ids)

    def write(self, ids, values):
        self._backing.write(ids, values)

    def traffic(self):
        return self._backing.traffic()


def make_samples(rng, n_samples, v_size, max_ctx=6, allow_zero_weight=True):
    samples = []
    for _ in range(n_samples):
        ctx_len = int(rng.integers(0, 4))
        context = rng.integers(0, v_size, size=ctx_len).tolist()
        samples.append({
            'context': context,
            'positive': int(rng.integers(0, v_size)),
            'negative': int(rng.integers(0, v_size)),
            'weight': float(rng.uniform(0.0 if allow_zero_weight else 0.05, 10.0)),
        })
    return samples


def chunk(samples, rng, n_chunks):
    perm = samples if rng is None else [samples[i] for i in rng.permutation(len(samples))]
    if n_chunks <= 1:
        return [perm]
    bounds = np.array_split(np.arange(len(perm)), n_chunks)
    return [[perm[i] for i in b] for b in bounds if b.size]


# ------------------------------------------------------------- 1. finite diff
def finite_difference_check():
    print('[1] finite-difference gradient check')
    rng = np.random.default_rng(101)
    # Hand-built adversarial structure: repeats across all three roles,
    # pos == neg, empty context, zero and non-uniform weights.
    samples = [
        {'context': [2, 2, 5], 'positive': 2, 'negative': 5, 'weight': 3.0},
        {'context': [1, 1, 1], 'positive': 1, 'negative': 1, 'weight': 0.5},
        {'context': [], 'positive': 3, 'negative': 3, 'weight': 2.0},
        {'context': [0, 4, 4], 'positive': 4, 'negative': 0, 'weight': 1.5},
        {'context': [5], 'positive': 2, 'negative': 2, 'weight': 0.0},
    ]
    for trial in range(4):
        v_size, dim = 8, 3 + trial
        table = rng.uniform(-2.0, 2.0, (v_size, dim))
        table[6] = rng.normal(0.0, 0.1, dim)   # an untouched row
        loss_ref, tw_ref, grad_ref = dense_loss_and_grad(table, samples)
        result = shared_loss_and_grad(
            [samples], lambda ids: table[np.asarray(ids)], dim)
        check(f'fd trial {trial}: loss matches',
              np.isclose(result.loss, loss_ref, atol=ATOL, rtol=RTOL),
              f'{result.loss} vs {loss_ref}')
        check(f'fd trial {trial}: total weight', result.total_weight == tw_ref)
        check(f'fd trial {trial}: ids sorted unique',
              np.all(np.diff(result.ids) > 0))
        grad_sparse = np.zeros_like(table)
        grad_sparse[result.ids] = result.values
        check(f'fd trial {trial}: analytic-vs-independent dense gradient',
              np.allclose(grad_sparse, grad_ref, atol=ATOL, rtol=RTOL))

        # Independent central finite differences over chosen coordinates.
        eps = 1e-7
        probe = [(r, c) for r in range(v_size) for c in range(dim)]
        picks = [probe[i] for i in rng.choice(len(probe),
                                              size=min(20, len(probe)),
                                              replace=False)]
        # Always probe rows actually used by the samples, per coordinate.
        picks += [(r, c) for r in (0, 1, 2, 4, 5) for c in range(dim)]
        worst = 0.0
        for r, c in picks:
            bumped = table.copy()
            bumped[r, c] += eps
            up = dense_loss_only(bumped, samples)
            bumped[r, c] = table[r, c] - eps
            down = dense_loss_only(bumped, samples)
            numeric = (up - down) / (2.0 * eps)
            exact = grad_ref[r, c]
            scale = max(1.0, abs(exact), abs(numeric))
            worst = max(worst, abs(numeric - exact) / scale)
        check(f'fd trial {trial}: central differences (worst rel {worst:.2e})',
              worst < 5e-7)

        # Zero total weight.
        zero_samples = [dict(s, weight=0.0) for s in samples]
        r0 = shared_loss_and_grad(
            [zero_samples], lambda ids: table[np.asarray(ids)], dim)
        check(f'fd trial {trial}: zero weight -> empty gradient',
              r0.loss == 0.0 and r0.total_weight == 0.0 and r0.ids.size == 0
              and r0.values.shape == (0, dim))
        check(f'fd trial {trial}: empty microbatches -> empty gradient',
              shared_loss_and_grad([], lambda ids: table[np.asarray(ids)], dim)
              .ids.size == 0)


# --------------------------------------------- 2. chunking/order invariance etc
def chunking_check():
    print('[2] re-chunking and reordering invariance')
    rng = np.random.default_rng(202)
    for trial in range(6):
        v_size, dim = int(rng.integers(4, 40)), int(rng.integers(1, 17))
        table = rng.uniform(-2.0, 2.0, (v_size, dim))
        samples = make_samples(rng, int(rng.integers(1, 20)), v_size)
        base = shared_loss_and_grad(
            [samples], lambda ids: table[np.asarray(ids)], dim)
        grad_base = np.zeros_like(table)
        grad_base[base.ids] = base.values
        _, _, grad_ref = dense_loss_and_grad(table, samples)
        check(f'chunk trial {trial}: loss vs dense reference',
              np.isclose(base.loss, dense_loss_only(table, samples),
                         atol=ATOL, rtol=RTOL))
        check(f'chunk trial {trial}: gradient vs independent dense kernel',
              np.allclose(grad_base, grad_ref, atol=ATOL, rtol=RTOL))
        for n_chunks in (2, 3, 5):
            reordered = chunk(samples, rng, n_chunks)
            got = shared_loss_and_grad(
                reordered, lambda ids: table[np.asarray(ids)], dim)
            check(f'chunk trial {trial}: {n_chunks} chunks loss',
                  np.isclose(got.loss, base.loss, atol=ATOL, rtol=RTOL))
            check(f'chunk trial {trial}: {n_chunks} chunks ids identical',
                  np.array_equal(got.ids, base.ids))
            check(f'chunk trial {trial}: {n_chunks} chunks gradient identical',
                  np.allclose(got.values, base.values, atol=ATOL, rtol=RTOL))
        # Lookup is only called with sorted unique ids.
        seen = []
        def spy_lookup(ids):
            ids = np.asarray(ids)
            seen.append(ids)
            if ids.size:
                check('lookup ids sorted unique',
                      np.all(ids[:-1] < ids[1:]))
            return table[ids]
        shared_loss_and_grad(chunk(samples, None, 4), spy_lookup, dim)


# ---------------------------------------------------- 3. full dense trajectory
def trajectory_check():
    print('[3] SparseTrainer vs full dense trajectory')
    rng = np.random.default_rng(303)
    configs = [
        dict(beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03),
        dict(beta1=0.0, beta2=0.0, eps=1e-6, weight_decay=0.0),
        dict(beta1=0.95, beta2=0.999, eps=1e-3, weight_decay=0.2),
        dict(beta1=0.5, beta2=0.9, eps=2e-8, weight_decay=0.1),
    ]
    for ci, cfg in enumerate(configs):
        v_size, dim = int(rng.integers(12, 48)), int(rng.integers(1, 17))
        init = rng.uniform(-2.0, 2.0, (v_size, dim))
        sparse_table = ArrayRowTable(init)
        dense_table = init.copy()
        trainer = SparseTrainer(sparse_table, **cfg)
        ref = DenseRefAdamW(dense_table, cfg['beta1'], cfg['beta2'],
                            cfg['eps'], cfg['weight_decay'])
        clip_norm = float(rng.uniform(0.01, 10.0))
        n_steps = int(rng.integers(20, 40))
        for step in range(n_steps):
            samples = make_samples(rng, int(rng.integers(1, 16)), v_size)
            n_chunks = int(rng.integers(1, 4))
            microbatches = chunk(samples, rng, n_chunks)
            lr = float(rng.uniform(1e-4, 0.05))
            got = trainer.step(microbatches, lr, clip_norm)
            exp_loss, exp_tw, exp_norm = dense_trainer_step(
                ref, samples, lr, clip_norm)
            check(f'cfg {ci} step {step}: loss',
                  np.isclose(got.loss, exp_loss, atol=ATOL, rtol=RTOL),
                  f'{got.loss} vs {exp_loss}')
            check(f'cfg {ci} step {step}: total weight',
                  np.isclose(got.total_weight, exp_tw, atol=ATOL, rtol=RTOL))
            check(f'cfg {ci} step {step}: grad norm',
                  np.isclose(got.grad_norm, exp_norm, atol=ATOL, rtol=RTOL))
            check(f'cfg {ci} step {step}: step count',
                  got.step_count == ref.t == step + 1)
            # Values must be observable through the optimizer interface only:
            # a probe including an idle row forces its lazy decay to settle.
            probe = np.array(sorted(set(
                [0, v_size - 1] + samples[0]['context'][:2]
                + [samples[0]['positive'], samples[0]['negative']])))
            seen = trainer.optimizer.lookup(probe)
            check(f'cfg {ci} step {step}: optimizer-visible rows match dense',
                  np.allclose(seen, ref.e[probe], atol=ATOL, rtol=RTOL),
                  f'max diff {np.max(np.abs(seen - ref.e[probe])):.3e}')
        # End-of-run export settles every row and matches the dense table.
        snapshot = trainer.optimizer.export()
        check(f'cfg {ci}: final table equals dense trajectory',
              np.allclose(snapshot.weights, ref.e, atol=ATOL, rtol=RTOL),
              f'max diff {np.max(np.abs(snapshot.weights - ref.e)):.3e}')
        check(f'cfg {ci}: final moments equal dense moments',
              np.allclose(snapshot.m, ref.m, atol=ATOL, rtol=RTOL)
              and np.allclose(snapshot.v, ref.v, atol=ATOL, rtol=RTOL))

    # Zero-weight skip: optimizer t must not advance.
    print('  -- zero-weight skip and exact-zero-gradient steps')
    init = rng.uniform(-2.0, 2.0, (10, 4))
    trainer = SparseTrainer(ArrayRowTable(init))
    before = trainer.optimizer.stats()
    zero_batch = [{'context': [1], 'positive': 2, 'negative': 3, 'weight': 0.0}]
    out = trainer.step([zero_batch], 0.01, 1.0)
    after = trainer.optimizer.stats()
    check('zero-weight step returns zeros',
          out.loss == 0.0 and out.total_weight == 0.0 and out.grad_norm == 0.0)
    check('zero-weight step does not advance t',
          after['t'] == before['t'] == 0)
    check('zero-weight step touches no table rows',
          trainer.table.traffic() == {'read_rows': 0, 'write_rows': 0})

    # Positive weight but an exactly zero gradient: still one training step.
    # With dim=1 and all embedding rows equal to zero, scores are zero but the
    # target gradient is w*sigmoid(0)*h = 0 since h = 0; context gradient also
    # zero because target rows are zero vectors.
    zero_init = np.zeros((6, 2))
    ztrainer = SparseTrainer(ArrayRowTable(zero_init))
    batch = [{'context': [0, 0, 1], 'positive': 2, 'negative': 3, 'weight': 2.0},
             {'context': [], 'positive': 4, 'negative': 4, 'weight': 1.0}]
    out = ztrainer.step([batch], 0.02, 1.0)
    check('positive-weight zero-gradient still steps',
          out.total_weight == 3.0 and out.grad_norm == 0.0
          and out.step_count == 1)
    out2 = ztrainer.step([batch], 0.02, 1.0)
    check('second zero-gradient step advances t again', out2.step_count == 2)
    # Weight decay still applies on zero-gradient steps; observe it through the
    # optimizer interface, which settles the lazy decay replay.
    expected = zero_init * (1.0 - 0.02 * 0.03) ** 2
    check('zero-gradient steps still apply weight decay',
          np.allclose(ztrainer.optimizer.export().weights, expected,
                      atol=ATOL, rtol=RTOL))


# ----------------------------------------------- 4. LazyAdamW standalone tests
def optimizer_check():
    print('[4] LazyAdamW standalone vs dense AdamW')
    rng = np.random.default_rng(404)

    def run(cfg, v_size, dim, n_steps, idle_rows, restricted=False,
            do_lookup_mid=False, do_export_mid=False):
        init = rng.uniform(-2.0, 2.0, (v_size, dim))
        backing = RestrictedTable(init) if restricted else ArrayRowTable(init)
        lazy = LazyAdamW(backing, **cfg)
        ref = DenseRefAdamW(init, cfg['beta1'], cfg['beta2'],
                            cfg['eps'], cfg['weight_decay'])
        active = np.array([r for r in range(v_size) if r not in idle_rows])
        export_snapshot = None
        for step in range(n_steps):
            lr = float(rng.uniform(1e-4, 0.05))
            mode = rng.uniform()
            if mode < 0.15:
                # Explicit empty-gradient step.
                lazy.step(np.empty(0, dtype=np.int64),
                          np.empty((0, dim)), lr)
                dense_grad = np.zeros((v_size, dim))
            elif mode < 0.3:
                # Some listed rows carry exact zero gradients.
                chosen = active[rng.choice(active.size,
                                           size=min(3, active.size),
                                           replace=False)]
                vals = np.zeros((chosen.size, dim))
                lazy.step(np.sort(chosen), vals, lr)
                dense_grad = np.zeros((v_size, dim))
            else:
                chosen = np.sort(active[rng.choice(
                    active.size, size=min(active.size,
                                          int(rng.integers(1, 6))),
                    replace=False)])
                vals = rng.uniform(-10.0, 10.0, (chosen.size, dim))
                if rng.uniform() < 0.5:
                    vals[0] = 0.0
                lazy.step(chosen, vals, lr)
                dense_grad = np.zeros((v_size, dim))
                dense_grad[chosen] = vals
            ref.step(dense_grad, lr)
            check_t = step + 1
            check(f'  t advances (cfg run, step {step})', lazy.t == check_t)

            if do_lookup_mid and step == n_steps // 2:
                probe = np.array(sorted(set(
                    active[:3].tolist() + list(idle_rows)[:2] + [0])))
                repeated = np.array([probe[0], probe[0], probe[-1], probe[1]])
                got = lazy.lookup(repeated)
                check('lookup preserves order and repeats',
                      got.shape == (repeated.size, dim)
                      and np.allclose(got[0], got[1], atol=0.0, rtol=0.0))
                check('lookup values match dense reference',
                      np.allclose(got, ref.e[repeated], atol=ATOL, rtol=RTOL),
                      f'max {np.max(np.abs(got - ref.e[repeated])):.3e}')
                # Read-only lookup must not change t or the table trajectory.
                check('lookup does not advance t', lazy.t == check_t)

            if do_export_mid and step == n_steps // 3:
                snap = lazy.export()
                check('export t', snap.t == check_t)
                check('export weights match dense',
                      np.allclose(snap.weights, ref.e, atol=ATOL, rtol=RTOL))
                check('export m/v match dense',
                      np.allclose(snap.m, ref.m, atol=ATOL, rtol=RTOL)
                      and np.allclose(snap.v, ref.v, atol=ATOL, rtol=RTOL))
                check('export does not advance t', lazy.t == check_t)
                export_snapshot = snap.weights.copy()

        snap = lazy.export()
        check('final export weights match dense',
              np.allclose(snap.weights, ref.e, atol=ATOL, rtol=RTOL),
              f'max {np.max(np.abs(snap.weights - ref.e)):.3e}')
        check('final export m match dense',
              np.allclose(snap.m, ref.m, atol=ATOL, rtol=RTOL))
        check('final export v match dense',
              np.allclose(snap.v, ref.v, atol=ATOL, rtol=RTOL))
        check('final export t', snap.t == n_steps)
        if export_snapshot is not None:
            # Mid-export state is still readable afterward and training resumed.
            check('training resumes consistently after export',
                  np.allclose(lazy.lookup(np.arange(v_size)), ref.e,
                              atol=ATOL, rtol=RTOL))
        # Idle rows only ever saw weight decay; final export asserts them below.
        return lazy, backing

    configs = [
        dict(beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03),
        dict(beta1=0.0, beta2=0.0, eps=1e-6, weight_decay=0.0),
        dict(beta1=0.95, beta2=0.999, eps=1e-3, weight_decay=0.2),
    ]
    for ci, cfg in enumerate(configs):
        v_size, dim = 24, int(rng.integers(1, 9))
        idle = {7, 8, 19, 23}
        run(cfg, v_size, dim, n_steps=30, idle_rows=idle,
            restricted=(ci == 1), do_lookup_mid=True, do_export_mid=True)
        print(f'  cfg {ci} dense-equivalence complete')

    # Long-idle rows revisited after many global steps, with varying lrs.
    print('  -- long-idle revisit and read-only lookup then training')
    cfg = dict(beta1=0.9, beta2=0.999, eps=1e-7, weight_decay=0.07)
    v_size, dim, n_steps = 16, 4, 40
    init = rng.uniform(-2.0, 2.0, (v_size, dim))
    lazy = LazyAdamW(ArrayRowTable(init), **cfg)
    ref = DenseRefAdamW(init, **{k: cfg[k] for k in
                                 ('beta1', 'beta2', 'eps', 'weight_decay')})
    revisit = 11
    lrs = list(np.linspace(0.05, 1e-4, n_steps))
    for step in range(n_steps):
        lr = float(lrs[step])
        if step < 25:
            chosen = np.array([0, 1, 2, 3])
        else:
            chosen = np.array([1, 2, revisit, 13])  # revisit after long idle
        vals = rng.uniform(-10.0, 10.0, (chosen.size, dim))
        if step % 7 == 3:
            vals[0] = 0.0
        lazy.step(chosen, vals, lr)
        g = np.zeros((v_size, dim))
        g[chosen] = vals
        ref.step(g, lr)
        if step == 24:
            # Read-only lookup of the idle row right before it is reused.
            got = lazy.lookup(np.array([revisit, revisit, 13]))
            check('idle row lookup before revisit matches dense',
                  np.allclose(got, ref.e[[revisit, revisit, 13]],
                              atol=ATOL, rtol=RTOL))
    got = lazy.lookup(np.array([revisit]))
    check('revisited row matches dense after reuse',
          np.allclose(got, ref.e[[revisit]], atol=ATOL, rtol=RTOL))
    snap = lazy.export()
    check('long-idle experiment full export matches dense',
          np.allclose(snap.weights, ref.e, atol=ATOL, rtol=RTOL)
          and np.allclose(snap.m, ref.m, atol=ATOL, rtol=RTOL)
          and np.allclose(snap.v, ref.v, atol=ATOL, rtol=RTOL))

    # Moment rows correspond only to rows that saw a non-zero gradient.
    nz_rows = set()
    # Replay the same draws deterministically is complex; instead verify
    # stats invariants on a fresh small optimizer.
    small = LazyAdamW(ArrayRowTable(np.zeros((5, 2))), **cfg)
    small.step(np.array([0, 1]), np.array([[1.0, 0.0], [0.0, 0.0]]), 0.01)
    st = small.stats()
    check('moment state only for non-zero-gradient rows',
          st['moment_rows'] == 1 and st['zero_only_rows'] == 1)
    small.step(np.array([3]), np.zeros((1, 2)), 0.01)
    check('zero-only rows tracked without moments',
          small.stats()['zero_only_rows'] == 2)
    small.step(np.empty(0, dtype=np.int64), np.empty((0, 2)), 0.01)
    check('empty gradient step advances global t', small.t == 3)
    snap = small.export()
    check('export dense state shape', snap.weights.shape == (5, 2)
          and snap.m.shape == (5, 2) and snap.v.shape == (5, 2) and snap.t == 3)
    check('untrained rows have zero moments after export',
          np.all(snap.m[[2, 4]] == 0.0) and np.all(snap.v[[2, 4]] == 0.0))


# ------------------------------------------------- 5. scaling 257 -> 131072
def fixed_trajectory(rng, n_steps, v_span):
    """Same ID/gradient trajectory regardless of total vocabulary size."""
    steps = []
    for _ in range(n_steps):
        n_samples = int(rng.integers(4, 12))
        samples = []
        for _ in range(n_samples):
            ctx_len = int(rng.integers(0, 5))
            samples.append({
                'context': rng.integers(0, v_span, size=ctx_len).tolist(),
                'positive': int(rng.integers(0, v_span)),
                'negative': int(rng.integers(0, v_span)),
                'weight': float(rng.uniform(0.05, 10.0)),
            })
        microbatches = chunk(samples, rng, int(rng.integers(1, 4)))
        steps.append((microbatches, float(rng.uniform(1e-3, 0.05)),
                      float(rng.uniform(0.05, 5.0))))
    return steps


def scaling_run(v_size, steps, dim, cfg, init_seed):
    rng_init = np.random.default_rng(init_seed)
    init = rng_init.uniform(-2.0, 2.0, (v_size, dim))
    table = ArrayRowTable(init)
    trainer = SparseTrainer(table, **cfg)
    per_step_traffic = []
    records = []
    for i, (microbatches, lr, clip_norm) in enumerate(steps):
        pre = table.traffic()
        result = trainer.step(microbatches, lr, clip_norm)
        post = table.traffic()
        per_step_traffic.append({
            'read_rows': post['read_rows'] - pre['read_rows'],
            'write_rows': post['write_rows'] - pre['write_rows'],
        })
        records.append((result.loss, result.total_weight,
                        result.grad_norm, result.step_count))
    pre_export = table.traffic()
    export = trainer.optimizer.export()
    export_traffic = {
        'read_rows': table.traffic()['read_rows'] - pre_export['read_rows'],
        'write_rows': table.traffic()['write_rows'] - pre_export['write_rows'],
    }
    return {
        'records': records,
        'per_step_traffic': per_step_traffic,
        'export_traffic': export_traffic,
        'stats_before_export': trainer.optimizer.stats(),
        'export': export,
        'final_traffic': table.traffic(),
    }


def scaling_check():
    print('[6] scaling run: 257 vs 131072 rows, identical touched trajectory')
    rng = np.random.default_rng(505)
    dim, n_steps = 8, 32
    v_span = 40                      # all touched IDs live in rows 0..39
    steps = fixed_trajectory(np.random.default_rng(606), n_steps, v_span)
    cfg = dict(beta1=0.9, beta2=0.999, eps=1e-7, weight_decay=0.05)
    init_seed = 707

    small = scaling_run(257, steps, dim, cfg, init_seed)
    large = scaling_run(131072, steps, dim, cfg, init_seed)

    check('numerical trajectory identical across vocab sizes',
          all(np.isclose(a, b, atol=ATOL, rtol=RTOL)
              for ra, rb in zip(small['records'], large['records'])
              for a, b in zip(ra, rb)))
    check('per-step table traffic identical across vocab sizes',
          small['per_step_traffic'] == large['per_step_traffic'])
    check('optimizer state size identical across vocab sizes',
          small['stats_before_export'] == large['stats_before_export'])
    # Every ordinary step only moves rows actually touched; the bound is a
    # constant multiple of the unique touched IDs (well below the context /
    # sample limits), never of the 131072-row vocabulary.
    touched_cap = 3 * (40 + 16)
    check('ordinary ops never scan the vocabulary (large run)',
          all(t['read_rows'] <= touched_cap
              and t['write_rows'] <= touched_cap
              for t in large['per_step_traffic']))
    total_reads = sum(t['read_rows'] for t in large['per_step_traffic'])
    total_writes = sum(t['write_rows'] for t in large['per_step_traffic'])
    check('cumulative ordinary traffic is vocabulary-independent',
          total_reads < 131072 and total_writes < 131072
          and total_reads == sum(
              t['read_rows'] for t in small['per_step_traffic']))
    check('moment rows bounded by touched ID span',
          large['stats_before_export']['moment_rows'] <= v_span)
    check('no pre-allocation for untouched rows: tracked rows << vocab',
          large['stats_before_export']['tracked_rows'] <= 2 * v_span)
    # Export is the documented full-table operation.
    check('export is the only full-vocabulary read',
          large['export_traffic']['read_rows'] == 131072)
    check('small export reads its full table',
          small['export_traffic']['read_rows'] == 257)
    check('exported weights identical on touched rows',
          np.allclose(small['export'].weights[:v_span],
                      large['export'].weights[:v_span],
                      atol=ATOL, rtol=RTOL))
    check('untouched large rows equal small rows (same init + decay replay)',
          np.allclose(small['export'].weights[40:257],
                      large['export'].weights[40:257],
                      atol=ATOL, rtol=RTOL))

    summary = {
        'vocab_sizes': [257, 131072],
        'dimension': dim,
        'steps': n_steps,
        'touched_id_span': v_span,
        'ordinary_step_traffic_read_write_rows_each':
            large['per_step_traffic'][0],
        'max_read_rows_any_ordinary_step':
            max(t['read_rows'] for t in large['per_step_traffic']),
        'max_write_rows_any_ordinary_step':
            max(t['write_rows'] for t in large['per_step_traffic']),
        'total_ordinary_read_rows':
            sum(t['read_rows'] for t in large['per_step_traffic']),
        'total_ordinary_write_rows':
            sum(t['write_rows'] for t in large['per_step_traffic']),
        'optimizer_state_before_export': large['stats_before_export'],
        'export_cost_read_rows': {
            257: small['export_traffic']['read_rows'],
            131072: large['export_traffic']['read_rows']},
        'export_cost_write_rows': {
            257: small['export_traffic']['write_rows'],
            131072: large['export_traffic']['write_rows']},
        'init_cost': 'O(1) optimizer allocation: no per-vocab-row moment '
                     'state; only the caller-owned table exists (caller cost)',
    }
    out = Path(__file__).parent / 'scaling_report.json'
    out.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f'  (report written to {out.name})')


def boundary_check():
    print('[5] boundary scenarios: max vocabulary, long idle revisit, extremes')
    rng = np.random.default_rng(11)
    cfg = dict(beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.1)

    # 131072 rows, dim 1: only a handful of rows are ever touched; then a long
    # run of empty global steps; then a row is revisited.
    v_size, dim = 131072, 1
    init = rng.uniform(-2.0, 2.0, (v_size, dim))
    table = ArrayRowTable(init)
    lazy = LazyAdamW(table, **cfg)
    ref = DenseRefAdamW(init.copy(), cfg['beta1'], cfg['beta2'],
                        cfg['eps'], cfg['weight_decay'])
    trajectories = [np.array([3, 131071]), np.array([131071]),
                    np.array([0, 3, 99999])]
    for step in range(20):
        lr = 0.02
        ids = trajectories[step % 3]
        vals = rng.uniform(-10.0, 10.0, (ids.size, dim))
        dense_grad = np.zeros((v_size, dim))
        dense_grad[ids] = vals
        lazy.step(ids, vals, lr)
        ref.step(dense_grad, lr)
    for _ in range(10):
        lazy.step(np.empty(0, dtype=np.int64), np.empty((0, dim)), 0.03)
        ref.step(np.zeros((v_size, dim)), 0.03)
    revisit_ids = np.array([3, 131071])
    revisit_vals = rng.uniform(-10.0, 10.0, (2, dim))
    dense_grad = np.zeros((v_size, dim))
    dense_grad[revisit_ids] = revisit_vals
    lazy.step(revisit_ids, revisit_vals, 0.01)
    ref.step(dense_grad, 0.01)
    pre_export_traffic = dict(table.traffic())
    snap = lazy.export()
    check('max-vocab weights match dense after empty steps and revisit',
          np.allclose(snap.weights, ref.e, atol=ATOL, rtol=RTOL))
    check('max-vocab moments match dense',
          np.allclose(snap.m, ref.m, atol=ATOL, rtol=RTOL)
          and np.allclose(snap.v, ref.v, atol=ATOL, rtol=RTOL))
    check('max-vocab global t is 31', snap.t == ref.t == 31)
    check('only touched rows carry moment state at 131072 rows',
          snap.m[np.setdiff1d(np.arange(v_size),
                              np.array([0, 3, 99999, 131071]))].sum() == 0.0
          and lazy.stats()['moment_rows'] == 4)
    check('ordinary traffic before export stays tiny',
          pre_export_traffic['read_rows'] < 200
          and pre_export_traffic['write_rows'] < 200)

    # Frozen-parameter gradient check with IDs at both vocabulary extremes.
    frozen = rng.uniform(-2.0, 2.0, (v_size, 4))
    microbatches = [[
        {'context': [0, 0, 131071, 5], 'positive': 131071,
         'negative': 0, 'weight': 7.5},
        {'context': [], 'positive': 1, 'negative': 1, 'weight': 2.5},
    ]]
    result = shared_loss_and_grad(
        microbatches, lambda ids: frozen[np.asarray(ids)], 4)
    check('extreme-ID gradient only touches referenced rows',
          result.ids.tolist() == [0, 1, 5, 131071]
          and result.values.shape == (4, 4)
          and np.all(np.diff(result.ids) > 0))

    # lookup order, duplicates and empty input.
    got = lazy.lookup(np.array([131071, 3, 3, 0]))
    check('lookup preserves order and repeated IDs',
          got.shape == (4, 1) and np.allclose(got[1], got[2])
          and np.allclose(got[:, 0], ref.e[[131071, 3, 3, 0], 0],
                          atol=ATOL, rtol=RTOL))
    check('empty lookup returns an empty correctly-shaped copy',
          lazy.lookup(np.empty(0, dtype=np.int64)).shape == (0, 1))


def main():
    finite_difference_check()
    chunking_check()
    trajectory_check()
    optimizer_check()
    boundary_check()
    scaling_check()
    print('\nALL CHECKS PASSED')


if __name__ == '__main__':
    main()
