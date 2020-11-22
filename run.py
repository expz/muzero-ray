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
from muzero.muzero import ATARI_DEFAULT_CONFIG
from muzero.trainer import MuZeroTrainer


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
      'env': 'BreakoutNoFrameskip-MuZero-v1',
      'action_type': 'atari',
      'num_workers': 3,
      'num_gpus': 1,
      'num_gpus_per_worker': 0,
      'log_level': 'WARNING',
      'learning_starts': 1024,
      'timesteps_per_iteration': 1024,
      'buffer_size': 65536,
      'optimizer': {
          'num_replay_buffer_shards': 1,
          'debug': False,
      },
  }
  config = update_config(ATARI_DEFAULT_CONFIG, config)

  register_muzero_env('BreakoutNoFrameskip-v4', 'BreakoutNoFrameskip-MuZero-v1')

  try:
      ray.shutdown()
  except:
      pass

  #ray.init(local_mode=True)
  ray.init(num_cpus=6, num_gpus=1)
  try:
      tune.run(
          MuZeroTrainer,
          local_dir='results',
          name='breakout',
          stop={'training_iteration': 4000},
          config=config,
          checkpoint_freq=16,
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
    parser.add_argument(
        '--config',
        type=str,
        help='path to JSON config file to override defaults'
    )
    args = parser.parse_args()

    main(args)