# Learning to play Atari

Unfortunately, while the training will run, the model does not learn. There is apparently an undiscovered bug or bad hyperparameter.

## Setup

```
make venv
```

## Run

```
source venv/bin/activate
python run.py --logdir ./results --loglevel error breakout
```

Ray dashboard will be available at localhost:8625.

## Tensorboard

```
source venv/bin/activate
tensorboard --logdir ./results
```
