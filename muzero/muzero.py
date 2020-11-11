import collections
import copy
import ray
from ray.actor import ActorHandle
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.utils.actors import create_colocated
from ray.rllib.execution.common import STEPS_TRAINED_COUNTER, _get_shared_metrics, _get_global_vars
from ray.rllib.execution.concurrency_ops import Concurrently, Enqueue, Dequeue
from ray.rllib.agents.dqn.learner_thread import LearnerThread
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.types import SampleBatchType
import tensorflow as tf
from typing import Tuple

from muzero.replay_buffer import ReplayActor, PRIO_WEIGHTS

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


# Update worker weights as they finish generating experiences.
class BroadcastUpdateLearnerWeights:
    """
    Adapted from https://github.com/ray-project/ray/blob/5acd3e66ddc1d7a1af6590567fcc3df95169d8a2/rllib/agents/impala/impala.py#L177
    """
    def __init__(self, learner_thread, workers, broadcast_interval):
        self.learner_thread = learner_thread
        self.steps_since_broadcast = collections.defaultdict(int)
        self.broadcast_interval = broadcast_interval
        self.workers = workers
        self.weights = workers.local_worker().get_weights()

    def __call__(self, item):
        actor, batch = item
        self.steps_since_broadcast[actor] += 1
        if (self.steps_since_broadcast[actor] >= self.broadcast_interval
                and self.learner_thread.weights_updated):
            self.weights = ray.put(self.workers.local_worker().get_weights())
            self.steps_since_broadcast[actor] = 0
            self.learner_thread.weights_updated = False
            # Update metrics.
            metrics = _get_shared_metrics()
            metrics.counters["num_weight_broadcasts"] += 1
        actor.set_weights.remote(self.weights, _get_global_vars())
        # Also update global vars of the local worker.
        self.workers.local_worker().set_global_vars(_get_global_vars())

def validate_config(config):
    # Update effective batch size to include n-step
    adjusted_batch_size = max(config["rollout_fragment_length"],
                              config.get("n_step", 1))
    config["rollout_fragment_length"] = adjusted_batch_size

    if config.get("prioritized_replay"):
        if config["multiagent"]["replay_mode"] == "lockstep":
            raise ValueError("Prioritized replay is not supported when "
                             "replay_mode=lockstep.")
        elif config["replay_sequence_length"] > 1:
            raise ValueError("Prioritized replay is not supported when "
                             "replay_sequence_length > 1.")

def calculate_rr_weights(config):
    if not config["training_intensity"]:
        return [1, 1]
    # e.g., 32 / 4 -> native ratio of 8.0
    native_ratio = (
        config["train_batch_size"] / config["rollout_fragment_length"])
    # Training intensity is specified in terms of
    # (steps_replayed / steps_sampled), so adjust for the native ratio.
    weights = [1, config["training_intensity"] / native_ratio]
    return weights

# workers: WorkerSet, config: dict
def mu_zero_learner_thread_execution_plan(workers, config):
    # Create a number of replay buffer actors.
    num_replay_buffer_shards = config["optimizer"]["num_replay_buffer_shards"]

    tensor_spec = (
        tf.TensorSpec((96, 96, 4), tf.float32, SampleBatch.CUR_OBS),
        tf.TensorSpec((1,), tf.int32, SampleBatch.EPS_ID),
        tf.TensorSpec((1,), tf.int32, SampleBatch.UNROLL_ID),
        tf.TensorSpec((1,), tf.int32, SampleBatch.ACTIONS),
        tf.TensorSpec((4,), tf.float32, SampleBatch.ACTION_DIST_INPUTS),
        tf.TensorSpec((1,), tf.float32, SampleBatch.ACTION_PROB),
        tf.TensorSpec((1,), tf.float32, SampleBatch.ACTION_LOGP),
        tf.TensorSpec((1,), tf.bool, SampleBatch.DONES),
        tf.TensorSpec((1,), tf.float32, SampleBatch.REWARDS),
        tf.TensorSpec((5, 4), tf.float32, 'rollout_policies'),
        tf.TensorSpec((5,), tf.float32, 'rollout_rewards'),
        tf.TensorSpec((5,), tf.float32, 'rollout_values'),
    )

    # There's no way to control where the ReplayActors are located in a multi-node setup, so
    # this function spins up more and more actors until it finds enough
    # which are on the local host.
    replay_actors = create_colocated(ReplayActor, [
        tensor_spec,
        num_replay_buffer_shards,
        config["learning_starts"],
        config["buffer_size"],
        config["train_batch_size"],
        config["prioritized_replay_alpha"],
        config["prioritized_replay_beta"],
        config["prioritized_replay_eps"],
        config["multiagent"]["replay_mode"],
        config["replay_sequence_length"],
    ], num_replay_buffer_shards)
    
    # Start the learner thread on the local host (where the ReplayActors are located).
    learner_thread = LearnerThread(
        workers.local_worker())
        #minibatch_buffer_size = config['minibatch_buffer_size'],
        #num_sgd_iter=config["num_sgd_iter"],
        #learner_queue_size=config["learner_queue_size"],
        #learner_queue_timeout=config["learner_queue_timeout"])
    learner_thread.start()

    # Update experience priorities post learning.
    def update_prio_and_stats(item: Tuple[ActorHandle, dict, int]) -> None:
        actor, prio_dict, count = item
        #print('actor', type(actor), actor)
        #print('prio_dict:', type(prio_dict), prio_dict)
        #print('count:', type(count), count)
        actor.update_priorities.remote(prio_dict)
        metrics = _get_shared_metrics()
        # Manually update the steps trained counter since the learner thread
        # is executing outside the pipeline.
        metrics.counters[STEPS_TRAINED_COUNTER] += count
        metrics.timers["learner_dequeue"] = learner_thread.queue_timer
        metrics.timers["learner_grad"] = learner_thread.grad_timer
        metrics.timers["learner_overall"] = learner_thread.overall_timer
        return metrics
    
    # We execute the following steps concurrently:
    # (1) Generate rollouts and store them in our replay buffer actors. Update
    # the weights of the worker that generated the batch.
    rollouts = ParallelRollouts(
        workers,
        mode="async",
        num_async=config['max_sample_requests_in_flight_per_worker'])
    store_op = rollouts.for_each(StoreToReplayBuffer(actors=replay_actors))

    if workers.remote_workers():
        store_op = store_op.zip_with_source_actor() \
            .for_each(BroadcastUpdateLearnerWeights(
                learner_thread,
                workers,
                broadcast_interval=config["broadcast_interval"]))

    # (2) Read experiences from the replay buffer actors and send to the
    # learner thread via its in-queue.
    replay_op = Replay(actors=replay_actors, num_async=4) \
        .zip_with_source_actor() \
        .for_each(Enqueue(learner_thread.inqueue))

        #.filter(WaitUntilTimestepsElapsed(config["learning_starts"])) \
    # (3) Get priorities back from learner thread and apply them to the
    # replay buffer actors.
    update_op = Dequeue(
        learner_thread.outqueue, check=learner_thread.is_alive) \
        .for_each(update_prio_and_stats)
    
    if config["training_intensity"]:
        # Execute (1), (2) with a fixed intensity ratio.
        rr_weights = calculate_rr_weights(config) + ["*"]
        merged_op = Concurrently(
            [store_op, replay_op, update_op],
            mode="round_robin",
            output_indexes=[2],
            round_robin_weights=rr_weights)
    else:
        # Execute (1), (2), (3) asynchronously as fast as possible. Only output
        # items from (3) since metrics aren't available before then.
        merged_op = Concurrently(
            [store_op, replay_op, update_op], mode="async", output_indexes=[2])

    def add_muzero_metrics(result):
        replay_stats = ray.get(replay_actors[0].stats.remote(
                config["optimizer"].get("debug")))
        result["info"].update({
            "learner_queue": learner_thread.learner_queue_size.stats(),
            "learner": copy.deepcopy(learner_thread.stats),
            "replay_shard_0": replay_stats,
        })
        return result

    return StandardMetricsReporting(merged_op, workers, config) \
        .for_each(add_muzero_metrics)

def get_policy_class(config):
    """
    This serves a dual purpose. It also prevents cyclic dependency between
    this module and muzero_tf_policy which import ATARI_DEFAULT_CONFIG from
    this module.
    """
    if config['framework'] == 'tf' or config['framework'] == 'tfe':
        from muzero.muzero_tf_policy import MuZeroTFPolicy
        return MuZeroTFPolicy
    else:
        raise NotImplementedError(
            f'Framework "{config["framework"]} not supported')

MuZeroTrainer = build_trainer(
    name="Muzero",
    default_policy=None,
    get_policy_class=get_policy_class,
    default_config=ATARI_DEFAULT_CONFIG,
    validate_config=validate_config,
    execution_plan=mu_zero_learner_thread_execution_plan)
