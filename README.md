# Learning to play Atari

This is an implementation of MuZero using the ray distributed processing library. It is meant to play the Atari game Breakout.

Unfortunately, while the training will run, the model does not learn. There is apparently an undiscovered bug or bad hyperparameter. Hyperparameters are set in `muzero/muzero.py` and overriden in `run.py`.

## Setup

```
make venv
```

## Test

To run tests, run this command from the root directory of the repository:
```
source venv/bin/activate
pytest
```

## Run for breakout

First, open up `run.py` and edit the ray configuration to have the amount of GPUs, CPUs and memory you would like. It is currently set to run on a 3 GPU machine with 12 cores and 64 GB of RAM.

Then from the root directory of this repo, run
```
source venv/bin/activate
TF_CPP_MIN_LOG_LEVEL=2 python run.py --logdir ./results --loglevel error breakout
```

Ray dashboard will be available at http://localhost:8265. If you are running the code on a server, then run `ssh -L 8265:localhost:8265 username@server` to make the dashboard available from your local machine.

Checkpoints will be available in `./results/breakout`.

## Tensorboard

```
source venv/bin/activate
tensorboard --logdir ./results
```

The tensorboard will be available at http://localhost:6006. If you are running the code on a server, then run `ssh -L 6006:localhost:6006 username@server` to make the tensorboard available from your local machine.

## Watch it play

To watch it play using a checkpoint, use `muzero-test-checkpoints.ipynb`. It is currently configured to run just on CPU, so it is slow. Change the ray configuration to run on GPU.
