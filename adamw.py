"""The existing dense, float64 AdamW implementation used by the training job."""
import numpy as np


class DenseAdamW:
    def __init__(self, weights, beta1=0.9, beta2=0.99, eps=1e-8, weight_decay=0.03):
        self.weights = np.array(weights, dtype=np.float64, copy=True)
        self.m = np.zeros_like(self.weights)
        self.v = np.zeros_like(self.weights)
        self.beta1, self.beta2 = beta1, beta2
        self.eps, self.weight_decay = eps, weight_decay
        self.step_count = 0

    def step(self, gradient, lr):
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != self.weights.shape or not np.isfinite(gradient).all():
            raise ValueError('gradient shape or value')
        if not np.isfinite(lr) or not 0 < lr <= 0.05:
            raise ValueError('learning rate')
        self.step_count += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient * gradient
        mhat = self.m / (1 - self.beta1 ** self.step_count)
        vhat = self.v / (1 - self.beta2 ** self.step_count)
        self.weights *= 1 - lr * self.weight_decay
        self.weights -= lr * mhat / (np.sqrt(vhat) + self.eps)
        return self.weights.copy()
