"""Lazy, row-access-only AdamW with the exact dense AdamW numerical trajectory.

The optimizer never scans the vocabulary on an ordinary operation. Rows are
touched only through the caller-owned table's ``read``/``write`` interface.
Rows that have not produced a non-zero gradient yet carry no moment state; when
they are first touched, only the decoupled weight decay accumulated over the
elapsed global steps is replayed in one batch. Rows with moment state replay the
elapsed steps one by one (an explicitly permitted strategy), which is required
because Adam's bias-corrected per-coordinate ratio has no exact closed form over
a missing interval. Every replay uses the global step counter for bias
correction, so the result is identical to applying dense AdamW to every row.
"""
from typing import NamedTuple

import numpy as np


class ExportState(NamedTuple):
    weights: np.ndarray
    m: np.ndarray
    v: np.ndarray
    t: int


class LazyAdamW:
    def __init__(self, table, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03):
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError('betas must be in [0, 1)')
        if eps <= 0.0 or weight_decay < 0.0:
            raise ValueError('eps must be positive and weight decay non-negative')
        self.table = table
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)

        self._t = 0
        # Per-step history, indexed by global step (step 1 at index 0).
        self._step_lr = []
        self._step_decay = []          # 1 - lr_s * weight_decay
        self._step_beta1_pow = []      # beta1 ** s
        self._step_beta2_pow = []      # beta2 ** s
        self._step_grads = []          # dict row_id -> gradient vector, only non-zero rows
        # Rows that have ever held non-zero moments: full numeric state.
        self._meta = {}
        # Rows touched only with zero gradient so far: last settled step only.
        self._zero_last = {}
        # All rows were settled through this step by the last full export (or 0).
        self._export_watermark = 0

    # ------------------------------------------------------------------ state
    @property
    def t(self):
        return self._t

    @property
    def step_count(self):
        return self._t

    @staticmethod
    def _as_ids(ids):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError('ids must be a 1-D array')
        return ids

    def _row_last(self, rid):
        """Global step through which row ``rid`` is currently settled."""
        entry = self._meta.get(rid)
        if entry is not None:
            return entry[2]
        last = self._zero_last.get(rid)
        if last is not None:
            return last
        return self._export_watermark

    def _flush(self, unique_ids, t_now):
        """Settle every requested row through global step ``t_now``.

        Only rows that are actually stale touch the table; each stale row is
        read once and, if it changed, written once. Returns a mapping
        rid -> settled weight vector for every requested row.
        """
        settled = {}
        stale_meta, stale_zero = [], []
        for rid in unique_ids:
            last = self._row_last(rid)
            if last >= t_now:
                continue
            entry = self._meta.get(rid)
            if entry is not None:
                stale_meta.append(rid)
            elif rid in self._zero_last:
                stale_zero.append(rid)
            else:
                # Never individually tracked: settled through the export
                # watermark; only decay applies over the remaining interval.
                stale_zero.append(rid)

        if stale_meta:
            values = self.table.read(np.asarray(stale_meta, dtype=np.int64))
            for rid, e in zip(stale_meta, values):
                _, _, last = self._meta[rid]
                e = np.array(e, dtype=np.float64, copy=True)
                for s in range(last + 1, t_now + 1):
                    grad = self._step_grads[s - 1].get(rid)
                    m, v, _ = self._meta[rid]
                    m = self.beta1 * m + (1.0 - self.beta1) * (
                        grad if grad is not None else 0.0)
                    if grad is None:
                        v = self.beta2 * v
                    else:
                        v = self.beta2 * v + (1.0 - self.beta2) * grad * grad
                    m_hat = m / (1.0 - self._step_beta1_pow[s - 1])
                    v_hat = v / (1.0 - self._step_beta2_pow[s - 1])
                    e *= self._step_decay[s - 1]
                    e -= self._step_lr[s - 1] * m_hat / (np.sqrt(v_hat) + self.eps)
                    self._meta[rid] = (m, v, s)
                settled[rid] = e
            self.table.write(np.asarray(stale_meta, dtype=np.int64),
                             np.stack([settled[r] for r in stale_meta]))

        if stale_zero:
            values = self.table.read(np.asarray(stale_zero, dtype=np.int64))
            out = []
            for rid, e in zip(stale_zero, values):
                last = self._row_last(rid)
                # Moment-free rows saw zero Adam updates on every missing step;
                # the product of per-step decay factors replays them exactly.
                decay = 1.0
                for s in range(last + 1, t_now + 1):
                    decay *= self._step_decay[s - 1]
                e = np.array(e, dtype=np.float64, copy=True) * decay
                settled[rid] = e
                out.append(e)
                self._zero_last[rid] = t_now
            self.table.write(np.asarray(stale_zero, dtype=np.int64), np.stack(out))

        return settled

    # --------------------------------------------------------------- operations
    def lookup(self, ids):
        """Return row values after the most recent global step.

        Order and repeated IDs are preserved. The returned array is a copy
        produced by the table's row interface.
        """
        ids = self._as_ids(ids)
        if ids.size == 0:
            return np.empty((0, self.table.shape[1]), dtype=np.float64)
        unique_ids, inverse = np.unique(ids, return_inverse=True)
        settled = self._flush(unique_ids, self._t)
        values = self.table.read(unique_ids)
        rows = np.empty((unique_ids.size, self.table.shape[1]), dtype=np.float64)
        for k, rid in enumerate(unique_ids):
            rid = int(rid)
            rows[k] = settled[rid] if rid in settled else values[k]
        return rows[inverse].copy()

    def step(self, ids, values, lr):
        """Apply one global AdamW step with the given per-row complete gradients.

        ``ids`` are unique; unlisted rows have a zero gradient this step. The
        global counter ``t`` always advances, including for empty or all-zero
        gradient steps.
        """
        ids = self._as_ids(ids)
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != ids.size:
            raise ValueError('values must have one row per id')
        if ids.size and np.any(ids[1:] <= ids[:-1]):
            raise ValueError('ids must be unique and ascending')
        if not np.isfinite(values).all():
            raise ValueError('gradient values must be finite')
        if ids.size and np.max(np.abs(values)) > 10.0 + 1e-12:
            raise ValueError('gradient values must be within [-10, 10]')
        if not np.isfinite(lr) or not 0.0 < lr <= 0.05:
            raise ValueError('learning rate must be in (0, 0.05]')

        t_now = self._t + 1
        self._t = t_now
        self._step_lr.append(float(lr))
        self._step_decay.append(1.0 - float(lr) * self.weight_decay)
        self._step_beta1_pow.append(self.beta1 ** t_now)
        self._step_beta2_pow.append(self.beta2 ** t_now)
        grad_map = {}
        self._step_grads.append(grad_map)

        if ids.size == 0:
            return

        # Settle requested rows through the previous step before applying the
        # new one, so every update starts from the correct dense trajectory.
        settled = self._flush(ids, t_now - 1)
        read_values = self.table.read(ids)
        current = {
            int(rid): settled[int(rid)] if int(rid) in settled else read_values[k]
            for k, rid in enumerate(ids)
        }
        decay = self._step_decay[-1]
        b1_corr = 1.0 - self._step_beta1_pow[-1]
        b2_corr = 1.0 - self._step_beta2_pow[-1]
        written_ids, written_values = [], []
        for rid, grad in zip(ids.tolist(), values):
            grad = grad.astype(np.float64, copy=False)
            old = current[rid]
            entry = self._meta.get(rid)
            if entry is None and not np.any(grad != 0.0):
                # Moment-free row: the Adam contribution stays exactly zero,
                # only decoupled weight decay touches the parameter.
                new = old * decay
                self._zero_last[rid] = t_now
            else:
                if entry is None:
                    m = (1.0 - self.beta1) * grad
                    v = (1.0 - self.beta2) * grad * grad
                    self._zero_last.pop(rid, None)
                else:
                    m_old, v_old, _ = entry
                    m = self.beta1 * m_old + (1.0 - self.beta1) * grad
                    v = self.beta2 * v_old + (1.0 - self.beta2) * grad * grad
                m_hat = m / b1_corr
                v_hat = v / b2_corr
                new = old * decay - float(lr) * m_hat / (np.sqrt(v_hat) + self.eps)
                self._meta[rid] = (m, v, t_now)
                if np.any(grad != 0.0):
                    grad_map[rid] = grad
            written_ids.append(rid)
            written_values.append(new)
        self.table.write(np.asarray(written_ids, dtype=np.int64),
                         np.stack(written_values))

    def export(self):
        """Settle the whole table through the current step without advancing t.

        This is an explicit full-vocabulary operation: it may read and write
        every row, and it allocates dense ``E``/``m``/``v`` arrays for the
        returned snapshot. Ordinary training never calls this path.
        """
        n_rows, dim = self.table.shape
        t_now = self._t
        weights = self.table.read(np.arange(n_rows, dtype=np.int64))
        m_full = np.zeros((n_rows, dim), dtype=np.float64)
        v_full = np.zeros_like(m_full)

        if t_now > self._export_watermark:
            # Rows with moment state: exact step-by-step replay.
            for rid, (m, v, last) in list(self._meta.items()):
                e = weights[rid]
                for s in range(last + 1, t_now + 1):
                    grad = self._step_grads[s - 1].get(rid)
                    m = self.beta1 * m + (1.0 - self.beta1) * (
                        grad if grad is not None else 0.0)
                    if grad is None:
                        v = self.beta2 * v
                    else:
                        v = self.beta2 * v + (1.0 - self.beta2) * grad * grad
                    m_hat = m / (1.0 - self._step_beta1_pow[s - 1])
                    v_hat = v / (1.0 - self._step_beta2_pow[s - 1])
                    e *= self._step_decay[s - 1]
                    e -= self._step_lr[s - 1] * m_hat / (np.sqrt(v_hat) + self.eps)
                weights[rid] = e
                m_full[rid] = m
                v_full[rid] = v
                self._meta[rid] = (m, v, t_now)

            # Moment-free rows: one batched decay product over each row's own
            # open interval. Rows only known through the export watermark use
            # the full interval; individually tracked zero-moment rows may have
            # been settled later than the watermark and must not decay twice.
            tracked = set(self._meta)
            for rid, last in list(self._zero_last.items()):
                if rid in tracked or last >= t_now:
                    continue
                row_decay = 1.0
                for s in range(last + 1, t_now + 1):
                    row_decay *= self._step_decay[s - 1]
                weights[rid] *= row_decay
                self._zero_last[rid] = t_now
            decay = 1.0
            for s in range(self._export_watermark + 1, t_now + 1):
                decay *= self._step_decay[s - 1]
            zero_rids = list(self._zero_last)
            untouched = np.ones(n_rows, dtype=bool)
            if tracked:
                untouched[list(tracked)] = False
            if zero_rids:
                untouched[zero_rids] = False
            weights[untouched] *= decay
            self.table.write(np.arange(n_rows, dtype=np.int64), weights)

        self._export_watermark = t_now
        self._zero_last.clear()
        return ExportState(weights=weights.copy(), m=m_full, v=v_full, t=t_now)

    def stats(self):
        """Sizes of the numeric state the optimizer actually stores.

        The caller-owned table itself is not counted. Moment rows hold two
        float64 vectors each; the step history holds one gradient vector per
        non-zero (row, step) pair; the remaining entries are bookkeeping
        integers/floats.
        """
        history_grad_rows = sum(len(grad_map) for grad_map in self._step_grads)
        return {
            't': self._t,
            'moment_rows': len(self._meta),
            'moment_vectors': 2 * len(self._meta),
            'zero_only_rows': len(self._zero_last),
            'tracked_rows': len(self._meta) + len(self._zero_last),
            'steps_remembered': len(self._step_lr),
            'history_grad_rows': history_grad_rows,
            'history_grad_vectors': history_grad_rows,
            'export_watermark': self._export_watermark,
        }
