from ray.rllib.agents.trainer import with_common_config


max_moves = 512

# Common config: https://docs.ray.io/en/latest/rllib-training.html#common-parameters

BOARD_DEFAULT_CONFIG = with_common_config({
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

ATARI_DEFAULT_CONFIG = with_common_config({
    'framework': 'tfe',
    'conv_filters': {
        'representation': [
            (1, 'conv', 128, (3, 3), (2, 2)),
            (2, 'res', 128, (3, 3), (1, 1)),
            (1, 'conv', 256, (3, 3), (2, 2)),
            (3, 'res', 256, (3, 3), (1, 1)),
            (1, 'avg_pool', None, (3, 3), (2, 2)),
            (2, 'res', 256, (3, 3), (1, 1)),
            (1, 'avg_pool', None, (3, 3), (2, 2)),
        ],
        'dynamics': [
            (1, 'conv', 256, (3, 3), (1, 1)),
            (16, 'res', 256, (3, 3), (1, 1)),
        ],
        'prediction': [
            (1, 'conv', 256, (3, 3), (1, 1)),
            #(19, 'res', 256, (3, 3), (1, 1)),
        ],
        'value_head': [
            (1, 'conv', 1, (1, 1), (1, 1)),
            (1, 'fc', 256, None, None),
        ],
        'policy_head': [
            (1, 'conv', 256, (3, 3), (1, 1)),
        ],
    },
    'action_type': 'atari',
    'preprocessor_pref': 'none',  # Prevent deepmind preprocessor from running
    'value_type': 'categorical',
    'value_max': 300,
    'reward_type': 'categorical',
    'reward_max': 300,
    'policy_type': 'fc',
    'input_steps': 32,
    'loss_steps': 5,
    'n_step': 10,
    'lr': 0.0005,
    'lr_schedule': None,
    'momentum': 0.9,
    'l2_reg': 1e-4,
    'gamma': 0.997,
    # The epsilon used in the formula for the invertible transform of model outputs.
    'scaling_epsilon': 0.001,
    'grad_clip': 40.0,
    'value_loss_weight': 0.25,  # See Reanalyze appendix
    'train_batch_size': 32,
    # The max number of observations the replay buffer can store.
    'buffer_size': 65536,
    # If set, this will fix the ratio of replayed from a buffer and learned
    # on timesteps to sampled from an environment and stored in the replay
    # buffer timesteps. Otherwise, replay will proceed as fast as possible.
    'training_intensity': None,
    # If you set a training_intensity, then this must be 0.
    'learning_starts': 1024,
    'rollout_fragment_length': 64,  # Number of steps of experience ot generate before saving batch
    'minibatch_buffer_size': 1,
    'num_sgd_iter': 1,
    'learner_queue_size': 16,
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
        'num_simulations': 10,
        'argmax_tree_policy': False,
        'puct_c1': 1.25,
        'puct_c2': 19652,
    },
    'optimizer': {
        'num_replay_buffer_shards': 2,
        'debug': False,
    },
})
