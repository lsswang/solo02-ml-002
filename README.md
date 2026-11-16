# Pairwise recommender baseline

Python 3.11 and NumPy 2.2.6. Run `python train.py` for the existing untied model.
`recommender.loss_and_grad` computes a weighted mean pairwise logistic loss with
independent input and output embedding tables. `DenseAdamW` follows the mathematical
dense update, including rows whose current gradient is zero. `ArrayRowTable` is a
small implementation of the caller-owned row access interface; a caller may provide
another object with only `shape`, `read(ids)` and `write(ids, values)`.

The existing entry point and dense numerical routines work for their stated scope.
The requested shared sparse training APIs and lazy optimizer are new capabilities;
see the task statement for their interfaces. No external data or GPU is needed.
