"""Pairwise logistic objectives: the legacy untied model and the shared-table model."""
from typing import NamedTuple, Sequence

import numpy as np


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


def loss_and_grad(batch, input_table, output_table):
    """Return weighted mean loss and dense gradients of two independent tables.

    Contexts are multisets: repeated IDs contribute repeatedly to their mean.
    Each example has exactly one positive and one negative target.
    """
    x, y = np.asarray(input_table), np.asarray(output_table)
    if x.ndim != 2 or y.shape != x.shape:
        raise ValueError('table shape')
    gx, gy = np.zeros_like(x), np.zeros_like(y)
    weighted_loss = 0.0
    total_weight = sum(example['weight'] for example in batch)
    if total_weight <= 0:
        return 0.0, gx, gy
    for example in batch:
        context = np.asarray(example['context'], dtype=np.int64)
        positive, negative, weight = example['positive'], example['negative'], example['weight']
        hidden = x[context].mean(axis=0) if len(context) else np.zeros(x.shape[1])
        pos_score, neg_score = hidden @ y[positive], hidden @ y[negative]
        weighted_loss += weight * (np.logaddexp(0.0, -pos_score) + np.logaddexp(0.0, neg_score))
        dpos, dneg = -weight * sigmoid(-pos_score), weight * sigmoid(neg_score)
        gy[positive] += dpos * hidden
        gy[negative] += dneg * hidden
        if len(context):
            dhidden = dpos * y[positive] + dneg * y[negative]
            np.add.at(gx, context, dhidden / len(context))
    return float(weighted_loss / total_weight), gx / total_weight, gy / total_weight


def clip_pair(grad_input, grad_output, limit):
    norm = np.sqrt(np.sum(grad_input * grad_input) + np.sum(grad_output * grad_output))
    scale = min(1.0, limit / norm) if norm else 1.0
    return grad_input * scale, grad_output * scale


class SharedLossResult(NamedTuple):
    loss: float
    total_weight: float
    ids: np.ndarray          # int64, sorted ascending, unique
    values: np.ndarray       # float64, shape (len(ids), dimension)


class _Example:
    """Flattened view of one weighted example with pre-resolved row indices."""
    __slots__ = ('weight', 'context_idx', 'pos_idx', 'neg_idx')

    def __init__(self, weight, context_idx, pos_idx, neg_idx):
        self.weight = weight
        self.context_idx = context_idx
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx


def shared_loss_and_grad(microbatches: Sequence[Sequence[dict]], lookup, dimension):
    """Weighted-mean loss and sparse gradient of the single shared embedding table.

    ``lookup(ids)`` must return the currently visible rows of the shared table for
    the given sorted, unique ID array, aligned with that array. Contexts are
    multisets: a repeated ID takes part in the mean repeatedly and accumulates
    gradient per occurrence. An empty context has the zero vector as its hidden
    state. Context, positive and negative positions reference the same parameter
    identity, so every gradient contribution to a row is summed.

    Microbatches are only a partitioning of one training step: no per-microbatch
    normalization, clipping or parameter update happens here, so re-chunking or
    reordering the same samples gives the same loss and gradient up to rounding.

    Returns a :class:`SharedLossResult`. When the total weight is zero the loss is
    zero and both gradient arrays are empty; no optimizer is required to call this
    function, which also serves fixed-parameter gradient checking.
    """
    dim = int(dimension)
    if dim <= 0:
        raise ValueError('dimension must be positive')

    flattened = []
    total_weight = 0.0
    needed = []
    for microbatch in microbatches:
        for example in microbatch:
            weight = float(example['weight'])
            total_weight += weight
            if weight > 0.0:
                context = np.asarray(example['context'], dtype=np.int64)
                pos = int(example['positive'])
                neg = int(example['negative'])
                needed.append(pos)
                needed.append(neg)
                if context.size:
                    needed.extend(context.tolist())
                flattened.append((weight, context, pos, neg, True))
            else:
                flattened.append((weight, None, 0, 0, False))

    if total_weight <= 0.0 or not flattened:
        return SharedLossResult(
            loss=0.0,
            total_weight=0.0,
            ids=np.empty(0, dtype=np.int64),
            values=np.empty((0, dim), dtype=np.float64),
        )

    ids = np.unique(np.asarray(needed, dtype=np.int64))
    rows = np.asarray(lookup(ids), dtype=np.float64)
    if rows.shape != (ids.size, dim):
        raise ValueError('lookup returned rows with an unexpected shape')
    position = {int(row_id): k for k, row_id in enumerate(ids)}
    grad = np.zeros((ids.size, dim), dtype=np.float64)

    resolved = []
    for weight, context, pos, neg, active in flattened:
        if not active:
            continue
        ctx_idx = [position[int(c)] for c in context] if context.size else None
        resolved.append(_Example(weight, ctx_idx, position[pos], position[neg]))

    weighted_loss = 0.0
    zero_hidden = np.zeros(dim, dtype=np.float64)
    for ex in resolved:
        if ex.context_idx is not None:
            hidden = rows[ex.context_idx].mean(axis=0)
            inv_count = 1.0 / len(ex.context_idx)
        else:
            hidden = zero_hidden
            inv_count = 0.0
        pos_row = rows[ex.pos_idx]
        neg_row = rows[ex.neg_idx]
        pos_score = float(hidden @ pos_row)
        neg_score = float(hidden @ neg_row)
        weighted_loss += ex.weight * (
            np.logaddexp(0.0, -pos_score) + np.logaddexp(0.0, neg_score)
        )
        dpos = -ex.weight * sigmoid(-pos_score)
        dneg = ex.weight * sigmoid(neg_score)
        grad[ex.pos_idx] += dpos * hidden
        grad[ex.neg_idx] += dneg * hidden
        if ex.context_idx is not None:
            dhidden = (dpos * pos_row + dneg * neg_row) * inv_count
            np.add.at(grad, ex.context_idx, dhidden)

    return SharedLossResult(
        loss=float(weighted_loss / total_weight),
        total_weight=float(total_weight),
        ids=ids,
        values=grad / total_weight,
    )


def global_l2_norm(values):
    """Global L2 norm of a stacked sparse gradient block."""
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.sum(values * values)))


def clip_global(values, limit):
    """Clip a stacked sparse gradient block by one shared, global L2 norm."""
    norm = global_l2_norm(values)
    scale = min(1.0, limit / norm) if norm > 0.0 else 1.0
    return np.asarray(values, dtype=np.float64) * scale, norm
