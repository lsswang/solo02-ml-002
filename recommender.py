"""Weighted pairwise logistic objective for the current untied embedding model."""
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
