# Pairwise recommender: shared sparse embeddings + lazy AdamW

Python 3.11 and NumPy 2.2.6, single-machine CPU, float64 throughout.

## Layout

| File | Role |
| --- | --- |
| `recommender.py` | Legacy untied `loss_and_grad`/`clip_pair` plus the new shared-table `shared_loss_and_grad`, `clip_global`, `SharedLossResult`. |
| `lazy_adamw.py` | `LazyAdamW`: row-access-only AdamW that reproduces the dense AdamW trajectory without scanning the vocabulary. `ExportState` is the `export()` snapshot. |
| `sparse_trainer.py` | `SparseTrainer.step(microbatches, lr, clip_norm)`: whole-step gradient, one global L2 clip, one optimizer update. |
| `row_table.py` | `ArrayRowTable`: example caller-owned table implementing only `shape`, `read(ids)` (returns a copy), `write(ids, values)`, plus observable traffic counters. |
| `adamw.py`, `train.py` | The original dense optimizer and the original two-table demo, unchanged and still runnable. |
| `verify.py` | Finite-difference, dense-trajectory, optimizer-interface, boundary and scaling checks. Writes `scaling_report.json`. |

## Setup and commands

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python train.py     # original untied two-table demo
.venv/bin/python verify.py    # full correctness + scaling verification
```

## Objective and shared gradient

For one example, with the hidden context mean `h` (repeated context IDs count
repeatedly; an empty context is the zero vector):

```
loss = softplus(-h . E[positive]) + softplus(h . E[negative])
```

The step loss is the weight-weighted sum divided by total weight. Context,
positive and negative positions reference the **same** row identity, so all
gradient contributions to a row are summed (positive may equal negative and
either may appear in the context).

`shared_loss_and_grad(microbatches, lookup, dimension)` performs one lookup of
the sorted-unique union of every referenced ID, then accumulates the complete
weighted-mean gradient of the whole step. It never normalizes, clips or updates
per microbatch, so re-chunking or reordering the same samples gives the same
loss and gradient within the stated tolerances. Zero total weight returns loss
0 and empty `ids`/`values`; no optimizer is needed, so it also serves fixed
parameter gradient checking.

## LazyAdamW semantics

`LazyAdamW(table, beta1, beta2, eps, weight_decay)` owns no embedding copy: the
caller-owned table is only accessed through `shape`/`read`/`write`.

* `step(ids, values, lr)`: `ids` unique and ascending, `values` the complete
  per-row gradients; unlisted rows have zero gradient. The global counter `t`
  advances on every call, including empty and all-zero gradient steps. Bias
  correction always uses the global `t` (never a per-row local step count), so
  the math is exactly equivalent to dense AdamW applied to every row.
* `lookup(ids)`: values after the latest global step, preserving order and
  repeated IDs; returns table-owned copies.
* `export()`: explicit full-vocabulary settle through the current `t` without
  adding a training step; returns dense `weights`, `m`, `v`, `t`.
* Zero-total-weight trainer steps skip everything and do not advance `t`. A
  positive-weight step with an exactly zero gradient is still a real step.

### Lazy settlement strategy and its costs

The optimizer keeps, per global step, only `lr`, the decay factor
`1 - lr*weight_decay`, the bias-correction powers and the sparse non-zero
gradients; per row it stores moment state `(m, v, last_settled_step)` only once
the row has seen a non-zero gradient. Rows touched solely with zero gradients
are bookkept with a single integer.

* **First touch of a never-trained row** (or any moment-free row): no Adam
  moments exist yet, so the open interval contains only decoupled weight decay,
  replayed in one batched product — O(interval) scalar multiplies, one row
  read/write.
* **Revisit of a row with moments**: Adam's bias-corrected per-coordinate ratio
  has no exact closed form over a missed interval, so the missing steps are
  replayed one by one from the stored per-step history. Cost per revisit is
  O(idle_steps x d) arithmetic plus one row read/write; it never touches other
  rows and never scans the table. Within the problem limits (at most 64 steps)
  this is small, and the global-step bias correction keeps it numerically
  identical to the dense trajectory.
* **Frequent revisits** are cheap: a row settled on the previous step has an
  empty replay interval, so each ordinary step costs O(unique touched rows)
  row reads/writes regardless of vocabulary size.
* **Long-interval revisits** cost one replay over the idle interval (at most
  64 steps here); state grows with rows that have produced non-zero gradients
  and with remembered step history, not with the vocabulary. No moment arrays
  are pre-allocated for untouched rows.
* `export()` and construction are the only operations permitted to be
  vocabulary-wide: `export()` reads and writes every row once and allocates the
  dense snapshot; optimizer construction itself allocates no per-row state.

## Scaling evidence

`verify.py` replays the identical touched-ID/gradient trajectory against a
257-row and a 131072-row table (all touched IDs stay inside rows 0..39) and
records real `ArrayRowTable` read/write counts and `LazyAdamW.stats()`. The
numerical trajectory, per-step traffic and optimizer state size are identical
across vocabulary sizes; ordinary steps move only touched rows (tens of rows,
not 131072), while only the explicit export performs full-table reads/writes.
See `scaling_report.json` after a run.

## Numerical contract

All values are finite float64; comparisons in `verify.py` use
`atol=2e-10, rtol=2e-9` (the implementation matches the dense reference to
rounding, typically ~1e-16 over the 64-step horizon). The reference dense
trajectory is an independent implementation in `verify.py`
(`DenseRefAdamW`, `dense_loss_and_grad`), and gradients are additionally
checked with an independent central finite-difference kernel.
