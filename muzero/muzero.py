import collections
import copy
import ray
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils.actors import create_colocated
from ray.rllib.execution.common import STEPS_TRAINED_COUNTER, _get_shared_metrics, _get_global_vars
from ray.rllib.execution.concurrency_ops import Concurrently, Enqueue, Dequeue
from ray.rllib.agents.dqn.learner_thread import LearnerThread
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import SimpleReplayBuffer, Replay, StoreToReplayBuffer, WaitUntilTimestepsElapsed, MixInReplay
from ray.rllib.execution.rollout_ops import ParallelRollouts, ConcatBatches
from ray.rllib.execution.train_ops import TrainOneStep, UpdateTargetNetwork
from ray.rllib.utils.types import SampleBatchType

from muzero.replay_buffer import ReplayActor

max_moves = 512

PRIO_WEIGHTS = 'weights'

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
            (19, 'res', 256, (3, 3), (1, 1)),
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
    'l2_reg': 1e-4,
    'gamma': 0.997,
    'grad_clip': 40.0,
    'train_batch_size': 256,
    'buffer_size': 131072,
    # number of sample batches to store for replay. The number of transitions
    # saved total will be (replay_buffer_num_slots * rollout_fragment_length).
    'replay_buffer_num_slots': 2048,
    # If set, this will fix the ratio of replayed from a buffer and learned
    # on timesteps to sampled from an environment and stored in the replay
    # buffer timesteps. Otherwise, replay will proceed as fast as possible.
    'training_intensity': None,
    # If you set a training_intensity, then this must be 0.
    'learning_starts': 0,
    'rollout_fragment_length': 64,  # Number of steps of experience ot generate before saving batch
    'minibatch_buffer_size': 1,
    'num_sgd_iter': 1,
    'learner_queue_size': 16,
    'learner_queue_timeout': 300,
    'broadcast_interval': 1,
    'max_sample_requests_in_flight_per_worker': 2,
    # set >0 to enable experience replay. Saved samples will be replayed with
    # a p:1 proportion to new data samples.
    'replay_proportion': 1,
    'prioritized_replay_alpha': 1,
    'prioritized_replay_beta': 1,
    'prioritized_replay_eps': 1e-6,
    'mcts': {
        'reset_q_bounds_per_node': True,
        'add_dirichlet_noise': False,
        'dirichlet_epsilon': None,
        'dirichlet_alpha': None,
        'num_simulations': 10,
        'argmax_tree_policy': False,
        'puct_c1': 1.25,
        'puct_c2': 19652,
    },
    'optimizer': {
        'num_replay_buffer_shards': 1,
        'debug': False,
    },
})


# Update worker weights as they finish generating experiences.
class UpdateWorkerWeights:
    def __init__(self, learner_thread, workers, max_weight_sync_delay):
        self.learner_thread = learner_thread
        self.workers = workers
        self.steps_since_update = collections.defaultdict(int)
        self.max_weight_sync_delay = max_weight_sync_delay
        self.weights = None

    # item: ("ActorHandle", SampleBatchType)
    def __call__(self, item):
        actor, batch = item
        self.steps_since_update[actor] += batch.count
        if self.steps_since_update[actor] >= self.max_weight_sync_delay:
            # Note that it's important to pull new weights once
            # updated to avoid excessive correlation between actors.
            if self.weights is None or self.learner_thread.weights_updated:
                self.learner_thread.weights_updated = False
                self.weights = ray.put(
                    self.workers.local_worker().get_weights())
            actor.set_weights.remote(self.weights, _get_global_vars())
            self.steps_since_update[actor] = 0
            # Update metrics.
            metrics = _get_shared_metrics()
            metrics.counters["num_weight_syncs"] += 1

# Update worker weights as they finish generating experiences.
class BroadcastUpdateLearnerWeights:
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

def mu_zero_execution_plan(workers, config):
    # Create a number of replay buffer actors.
    rollouts = ParallelRollouts(workers, mode='bulk_sync')
    
    replay_buffer = SimpleReplayBuffer(config['buffer_size'])
    
    store_op = rollouts \
            .for_each(StoreToReplayBuffer(local_buffer=replay_buffer))

    replay_op = Replay(local_buffer=replay_buffer) \
        .filter(WaitUntilTimestepsElapsed(config["learning_starts"])) \
        .combine(ConcatBatches(min_batch_size=config["train_batch_size"])) \
        .for_each(TrainOneStep(workers, num_sgd_iter=config["num_sgd_iter"]))

    train_op = Concurrently(
        [store_op, replay_op], mode="round_robin", output_indexes=[1])

    return StandardMetricsReporting(train_op, workers, config)

# workers: WorkerSet, config: dict
def mu_zero_learner_thread_execution_plan(workers, config):
    # Create a number of replay buffer actors.
    num_replay_buffer_shards = config["optimizer"]["num_replay_buffer_shards"]
    # There's no way to control where the ReplayActors are located in a multi-node setup, so
    # this function spins up more and more actors until it finds enough
    # which are on the local host.
    replay_actors = create_colocated(ReplayActor, [
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
    def update_prio_and_stats(item: ("ActorHandle", dict, int)):
        actor, prio_dict, count = item
        actor.update_priorities.remote(prio_dict)
        metrics = _get_shared_metrics()
        # Manually update the steps trained counter since the learner thread
        # is executing outside the pipeline.
        metrics.counters[STEPS_TRAINED_COUNTER] += count
        metrics.timers["learner_dequeue"] = learner_thread.queue_timer
        metrics.timers["learner_grad"] = learner_thread.grad_timer
        metrics.timers["learner_overall"] = learner_thread.overall_timer
    
    # We execute the following steps concurrently:
    # (1) Generate rollouts and store them in our replay buffer actors. Update
    # the weights of the worker that generated the batch.
    rollouts = ParallelRollouts(
        workers,
        mode="async",
        num_async=config['max_sample_requests_in_flight_per_worker'])
    store_op = rollouts.for_each(StoreToReplayBuffer(actors=replay_actors))

    """
    store_op = rollouts \
        .for_each(lambda batch: batch.decompress_if_needed()) \
        .for_each(MixInReplay(
            num_slots=config['replay_buffer_num_slots'],
            replay_proportion=config['replay_proportion'])) \
        .flatten() \
        .combine(ConcatBatches(min_batch_size=config['train_batch_size'])) \
        .for_each(Enqueue(learner_thread.inqueue))
    """

    if workers.remote_workers():
        store_op = store_op.zip_with_source_actor() \
            .for_each(BroadcastUpdateLearnerWeights(
                learner_thread,
                workers,
                broadcast_interval=config["broadcast_interval"]))

    """
    # Only need to update workers if there are remote workers.
    if workers.remote_workers():
        store_op = store_op.zip_with_source_actor() \
            .for_each(UpdateWorkerWeights(
                learner_thread, workers,
                max_weight_sync_delay=(
                    config["optimizer"]["max_weight_sync_delay"])
            ))
    """

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
    
    """
    .for_each(UpdateTargetNetwork(
        workers,
        config["target_network_update_freq"],
        by_steps_trained=True))
    """

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
