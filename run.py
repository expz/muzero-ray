#!/usr/bin/env python3
"""
Train a MuZero model.
"""

import argparse
import coloredlogs
import copy
import logging
import ray
from ray import tune

from muzero.env import register_muzero_env
from muzero.muzero import ATARI_DEFAULT_CONFIG, MuZeroTrainer


log = logging.getLogger(__name__)


def update_config(old_config, new_config):
  config = {**old_config}
  for k, v in new_config.items():
    if k in config and isinstance(v, dict):
      config[k] = update_config(config[k], v)
    else:
      config[k] = v
  return config


def main(args):
  coloredlogs.install(level=args.loglevel.upper())

  config = {
      'num_workers': 1,
      'log_level': 'DEBUG',
      'learning_starts': 0,
      'env': 'BreakoutNoFrameskip-MuZero-v1',
  }
  config = update_config(ATARI_DEFAULT_CONFIG, config)

  register_muzero_env('BreakoutNoFrameskip-v4', 'BreakoutNoFrameskip-MuZero-v1')

  try:
      ray.shutdown()
  except:
      pass

  ray.init(local_mode=True)
  try:
    tune.run(
      MuZeroTrainer,
      local_dir='results',
      name='breakout',
      stop={'training_iteration': 4},
      config=config,
      checkpoint_at_end=True)
  finally:
    ray.shutdown()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='train a MuZero algorithm on a specified game'
  )
  parser.add_argument(
    'game',
    type=str,
    choices=['tictactoe', 'atari'],
    help='the game to learn'
  )
  parser.add_argument(
    '--outfile',
    type=str,
    help='path where to save the trained model'
  )
  parser.add_argument(
    '--chk',
    type=str,
    help='path to checkpoint of model to load before training'
  )
  parser.add_argument(
    '--loglevel',
    type=str,
    choices=['debug', 'info', 'warning', 'error', 'critical'],
    default='info',
    help='level of log to print'
  )
  args = parser.parse_args()

  main(args)