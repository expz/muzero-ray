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
import tensorflow as tf

tf.get_logger().setLevel('WARNING')

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
        'num_workers': 4,
        'num_gpus': 1,
        'num_cpus_per_worker': 1,
        'num_gpus_per_worker': 0,
        'memory_per_worker': 4 * 1024**3,
        'object_store_memory_per_worker': 2 * 1024**3,
        'log_level': args.loglevel.upper(),
        'learning_starts': 256,
        'timesteps_per_iteration': 512,
        'buffer_size': 100000,
        'train_batch_size': 32,
        'replay_batch_size': 32,
        'optimizer': {
            'num_replay_buffer_shards': 1,
            'debug': False,
        },
    }
    config = update_config(ATARI_DEFAULT_CONFIG, config)

    if args.game == 'breakout':
        register_muzero_env('BreakoutNoFrameskip-v4', 'BreakoutNoFrameskip-MuZero-v1')

    checkpoint_freq = args.checkpoint_steps // config['timesteps_per_iteration']

    try:
        ray.shutdown()
    except:
        pass

    #ray.init(local_mode=True)
    ray.init(
        num_cpus=11,
        num_gpus=3,
        object_store_memory=20 * 1024**3,
        _redis_max_memory=5 * 1024**3,
        _memory=25 * 1024**3
    )
    try:
        if args.checkpoint:
            tune.run(
                MuZeroTrainer,
                local_dir=args.logdir,
                name=args.game,
                stop={'training_iteration': args.steps // config['timesteps_per_iteration']},
                config=config,
                checkpoint_freq=checkpoint_freq,
                checkpoint_at_end=True,
                restore=args.checkpoint)
        else:
            tune.run(
                MuZeroTrainer,
                local_dir=args.logdir,
                name=args.game,
                stop={'training_iteration': args.steps // config['timesteps_per_iteration']},
                config=config,
                checkpoint_freq=checkpoint_freq,
                checkpoint_at_end=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train a model using the MuZero algorithm on a specified game'
    )
    parser.add_argument(
        'game',
        type=str,
        choices=['breakout'],
        help='the game to learn'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default='results',
        help='path where to save the metrics and trained model'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='path to checkpoint of model to load before training.'
            ' this should be something like ./logdir/game/experiment/checkpoint-1/checkpoint-1'
    )
    parser.add_argument(
        '--loglevel',
        type=str,
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='warning',
        help='level of log to print'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=1000000,
        help='number of steps for which to train',
    )
    parser.add_argument(
        '--checkpoint-steps',
        type=int,
        default=10000,
        help='checkpoint every this number of steps',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='path to JSON config file to override defaults'
    )
    args = parser.parse_args()

    main(args)
