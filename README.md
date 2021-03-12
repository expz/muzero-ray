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

## Train on CartPole

### Configure resources if necessary

The current settings will request 4 GiB of RAM and 5 CPU cores. These can be changed in `run.py` by changing:
```
total_cpus = <TOTAL CPUS TO BE USED>
total_gpus = <TOTAL GPUS TO BE USED>
config = {
    'num_gpus': <NUM GPUS FOR LEARNER (AT MOST 1 IS SUPPORTED)>,
    'num_workers': <NUM WORKERS>,
    'num_cpus_per_worker': <NUM CPUS PER WORKER>,
    'num_gpus_per_worker': <NUM GPUS PER WORKER (CAN BE FRACTIONAL)>,
    'memory_per_worker': <MEMORY PER WORKER IN BYTES>,
    'object_store_memory_per_worker': <SHARED MEMORY PER WORKER IN BYTES>,
    ...
    'train_batch_size': <SET THIS TO CHANGE FROM DEFAULT 512>,
}
```
Please note that GPU does not currently offer much of a speed up, because the GPU usage has not been optimized.

### Run

From the root directory of the repository, run:
```
source venv/bin/activate
TF_CPP_MIN_LOG_LEVEL=2 python run.py cartpole
```
The ray dashboard will be available at http://localhost:8265. If you are running the code on a server, then log in using `ssh -L 8265:localhost:8265 username@server` to make the dashboard available from your local machine.

Checkpoints will be available in `./results/cartpole`.

## Train on Atari Breakout

As for CartPole, configure the resources if necessary. It is currently set to run on a 3 GPU machine with 12 cores and 64 GB of RAM.

Then from the root directory of this repo, run
```
source venv/bin/activate
TF_CPP_MIN_LOG_LEVEL=2 python run.py --logdir ./results --loglevel error breakout
```

The ray dashboard will be available at http://localhost:8265. If you are running the code on a server, then log in using `ssh -L 8265:localhost:8265 username@server` to make the dashboard available from your local machine.

Checkpoints will be available in `./results/breakout`.

## Tensorboard

During or after training, run from the root directory of the repository:
```
source venv/bin/activate
tensorboard --logdir ./results
```

The tensorboard will be available at http://localhost:6006. If you are running the code on a server, then log in using `ssh -L 6006:localhost:6006 username@server` to make the tensorboard available from your local machine.

## Watch it play

To watch it play using a checkpoint, use `muzero-test-checkpoints.ipynb`. It is currently configured to run just on CPU, so it is slow. Change the ray configuration to run on GPU.
