"""Small existing training entry point, with separate input and output tables."""
import json
from pathlib import Path
import numpy as np
from adamw import DenseAdamW
from recommender import loss_and_grad, clip_pair


class Trainer:
    def __init__(self, inputs, outputs):
        self.inputs, self.outputs = DenseAdamW(inputs), DenseAdamW(outputs)

    def step(self, batch, lr, clip_norm=1.0):
        loss, gx, gy = loss_and_grad(batch, self.inputs.weights, self.outputs.weights)
        gx, gy = clip_pair(gx, gy, clip_norm)
        self.inputs.step(gx, lr)
        self.outputs.step(gy, lr)
        return loss


def main():
    batch = json.loads((Path(__file__).parent / 'examples.json').read_text())
    rng = np.random.default_rng(4)
    trainer = Trainer(rng.normal(0, .2, (7, 3)), rng.normal(0, .2, (7, 3)))
    losses = [trainer.step(batch, .02) for _ in range(12)]
    print(json.dumps({'first_loss': losses[0], 'last_loss': losses[-1],
                      'steps': trainer.inputs.step_count}))


if __name__ == '__main__':
    main()
