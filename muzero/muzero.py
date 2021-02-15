max_moves = 512

DEFAULT_CONFIG = {
    # Number of CPUs to allocate for the trainer. Note: this only takes effect
    # when running in Tune. Otherwise, the trainer runs in the main program.
    'num_gpus': 1,
    'num_cpus_for_driver': 1,
    'num_cpus_per_worker': 1,
    'num_gpus_per_worker': 0,
    # Heap memory for the trainer process.
    'memory': 0,
    # Object store memory for the trainer process.
    'object_store_memory': 0,
    'memory_per_worker': 0,
    'object_store_memory_per_worker': 0,
    'custom_resources_per_worker': {},
    'multiagent': {
        'replay_mode': 'independent',
    },
    'timesteps_per_iteration': 1024,
    'min_iter_time_s': 0,
    'metrics_smoothing_episodes': 20,
    'collect_metrics_timeout': 180,
}


def config(conf: dict) -> dict:
    new_conf = {**DEFAULT_CONFIG}
    new_conf.update(conf)
    return new_conf


BOARD_DEFAULT_CONFIG = config({
    'framework': 'tfe',
    'conv_filters': {
        'representation': [
            (1, 'conv', 256, (3, 3), (1, 1)),
            (16, 'res', 256, (3, 3), (1, 1)),
        ],
        'dynamics': [
            (1, 'conv', 256, (3, 3), (1, 1)),
            (16, 'res', 256, (3, 3), (1, 1)),
        ],
        'prediction': [
            (1, 'conv', 256, (3, 3), (1, 1)),
            (19, 'res', 256, (3, 3), (1, 1)),
        ],
        'value_head': [
            (1, 'conv', 1, (1, 1), (1, 1)),
            (1, 'fc', 256, None, None),
        ],
        'policy_head': [
            (1, 'conv', 256, (3, 3), (1, 1)),
        ]
    },
    'action_type': 'board',
    'envs_per_worker': 2,
    'preprocessor_pref': 'none',  # Prevent deepmind preprocessor from running
    'board_shape': (9, 9),
    'value_type': 'scalar',
    'value_max': 1,
    'reward_type': 'scalar',
    'reward_max': 1,
    'policy_type': 'conv',  # Go uses 362 node fc layer
    'input_steps': 8,  # Chess uses last 100 states
    'loss_steps': 5,
    'n_step': max_moves,
    'l2_reg': 1e-4,
    'gamma': 1,
    'prioritized_replay_alpha': 0,  # uniform sampling
    'prioritized_replay_beta': 1,
    'prioritized_replay_eps': 1e-6,
    'buffer_size': 131072,
    # number of sample batches to store for replay. The number of transitions
    # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
    'replay_buffer_num_slots': 2048,
    'replay_sequence_length': max_moves,
    'train_batch_size': max_moves * 2,
    'rollout_fragment_length': max_moves,  # Number of steps of experience ot generate before saving batch
    'num_sgd_iter': 1,
    'multiagent': {
        'replay_mode': 'independent',
    },
    'mcts': {
        'reset_q_bounds_per_node': True,
        'add_dirichlet_noise': False,
        'dirichlet_epsilon': None,  # TODO: look me up in AlphaGo Zero paper
        'dirichlet_alpha': 0.3,  # 0.3 for chess, 0.15 for shogi, 0.03 for go
        'num_simulations': 10,
        'argmax_tree_policy': False,
        'puct_c1': 1.25,
        'puct_c2': 19652,
    },
    'optimizer': {
        'num_replay_buffer_shards': 1,
    },
})

ATARI_DEFAULT_CONFIG = config({
    'framework': 'tfe',
    'conv_filters': {
        'representation': [
            (1, 'conv', 128, (3, 3), (2, 2)),
            (2, 'res', 128, (3, 3), (1, 1)),
            (1, 'conv', 256, (3, 3), (2, 2)),
            (3, 'res', 256, (3, 3), (1, 1)),
            (1, 'avg_pool', None, (3, 3), (2, 2)),
            (3, 'res', 256, (3, 3), (1, 1)),
            (1, 'avg_pool', None, (3, 3), (2, 2)),
            (16, 'res', 256, (3, 3), (1, 1)),
        ],
        'dynamics': [
            (1, 'conv', 256, (3, 3), (1, 1)),
            (16, 'res', 256, (3, 3), (1, 1)),
        ],
        'reward': [
            (1, 'conv', 1, (1, 1), (1, 1)),
            (1, 'fc', 256, None, None),
        ],
        'prediction': [
        ],
        'value': [
            (1, 'conv', 1, (1, 1), (1, 1)),
            (1, 'fc', 256, None, None),
        ],
        'policy': [
            (1, 'conv', 2, (1, 1), (1, 1)),
        ],
    },
    'action_type': 'atari',
    'envs_per_worker': 1,
    'preprocessor_pref': 'none',  # Prevent deepmind preprocessor from running
    'value_type': 'categorical',
    'value_max': 300,
    'reward_type': 'categorical',
    'reward_max': 300,
    'policy_type': 'fc',
    'input_steps': 32,  # Number of frames per input
    'n_channels': 4,  # Number of channels per frame
    'loss_steps': 5,
    # The paper used 10, and 5 for the reanalyze version
    'n_step': 6,
    # The paper used 0.05 with batch size 1024
    'lr': 0.002,
    'lr_schedule': None,
    # The paper used 0.9 with batch size 1024
    'momentum': 0.9,
    # The paper used 1e-4 with batch size 1024
    # 'l2_reg': 4e-5,
    'l2_reg': 4e-6,
    'gamma': 0.997,
    # Apply invertible transform of value and reward model outputs.
    # The paper does this, but it is unclear whether I correctly implemented
    # it, so leave it off.
    'transform_outputs': False,
    # The epsilon used in the formula for the invertible transform of model outputs.
    'scaling_epsilon': 0.001,
    'grad_clip': 40.0,
    'value_loss_weight': 0.25,  # See Reanalyze appendix
    # The number of frames to generate before returning
    'replay_batch_size': 48,
    # The paper uses batch size of 1024
    'train_batch_size': 48,
    # The max number of observations the replay buffer can store.
    'buffer_size': 100000,
    # If set, this will fix the ratio of replayed from a buffer and learned
    # on timesteps to sampled from an environment and stored in the replay
    # buffer timesteps. Otherwise, replay will proceed as fast as possible.
    'training_intensity': None,
    # If you set a training_intensity, then this must be 0.
    'learning_starts': 512,
    # Shutdown and respawn workers after this many timesteps. Set to 0 to disable.
    'memory_reset_interval': 10000,
    # Deprecated. Set to batch size.
    'rollout_fragment_length': 48,
    'minibatch_buffer_size': 1,
    'num_sgd_iter': 1,
    'learner_queue_size': 8,
    'learner_queue_timeout': 60,
    'broadcast_interval': 1,
    'max_sample_requests_in_flight_per_worker': 2,
    'prioritized_replay_alpha': 1,
    'prioritized_replay_beta': 1,
    'prioritized_replay_eps': 1e-6,
    'mcts': {
        'reset_q_bounds_per_node': True,
        'add_dirichlet_noise': True,
        'dirichlet_epsilon': 0.25,
        'dirichlet_alpha': 0.25,
        # The paper used 50, but showed that it could work with as little as 7
        'num_simulations': 20,
        'argmax_tree_policy': False,
        'puct_c1': 1.25,
        'puct_c2': 19652,
    },
    'optimizer': {
        'num_replay_buffer_shards': 1,
        'debug': False,
    },
})
