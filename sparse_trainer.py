"""Shared-table sparse trainer wiring the shared objective to LazyAdamW."""
from typing import NamedTuple, Sequence

import numpy as np

from lazy_adamw import LazyAdamW
from recommender import clip_global, shared_loss_and_grad


class StepResult(NamedTuple):
    loss: float
    total_weight: float
    grad_norm: float
    step_count: int


class SparseTrainer:
    """Trains one shared embedding table through the row-access interface.

    One training step: compute the complete weighted-mean gradient over all
    microbatches against the currently visible shared parameters, clip the
    merged sparse gradient by one global L2 norm, then perform exactly one
    optimizer update. A zero-total-weight step is skipped entirely and does not
    advance the optimizer's global counter; a positive-weight step with an
    exactly zero gradient is still a real training step.
    """

    def __init__(self, table, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03):
        self.table = table
        self.optimizer = LazyAdamW(
            table, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay)
        self.dimension = int(table.shape[1])

    @property
    def step_count(self):
        return self.optimizer.step_count

    def step(self, microbatches: Sequence[Sequence[dict]], lr, clip_norm=1.0):
        if clip_norm <= 0.0:
            raise ValueError('clip_norm must be positive')
        result = shared_loss_and_grad(
            microbatches, self.optimizer.lookup, self.dimension)
        loss, total_weight, ids, grads = result

        if total_weight <= 0.0:
            # Whole step skipped: no optimizer step, t must not advance.
            return StepResult(loss=0.0, total_weight=0.0,
                              grad_norm=0.0, step_count=self.optimizer.step_count)

        clipped, grad_norm = clip_global(grads, clip_norm)
        # One update for the merged, globally clipped gradient.
        self.optimizer.step(ids, clipped, lr)
        return StepResult(loss=loss, total_weight=total_weight,
                          grad_norm=grad_norm,
                          step_count=self.optimizer.step_count)
