from collections import deque

import cv2
import gym
import numpy as np
import ray.tune as tune


class NoopResetEnv(gym.Wrapper):
    """
    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/env/atari_wrappers.py
    """
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class MaxAndSkipEnv(gym.Wrapper):
    """
    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/env/atari_wrappers.py
    """
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2, ) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class EpisodicLifeEnv(gym.Wrapper):
    """
    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/env/atari_wrappers.py
    """
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condtion for a few fr
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class FireResetEnv(gym.Wrapper):
    """
    From https://github.com/ray-project/ray/blob/ray-0.8.7/rllib/env/atari_wrappers.py
    """
    def __init__(self, env):
        """Take action on reset.
        For environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class ResizeFrame(gym.ObservationWrapper):
    def __init__(self, env, dim):
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 3),
            dtype=np.uint8)
    
    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

class FrameStackWithAction(gym.Wrapper):
    def __init__(self, env, k):
        raise NotImplementedError('FrameStackWithAction is an abstract base class')
    
    def _add_action(self, ob, action):
        """Add frame with constant value representing action. Notice the plus one."""
        action_frame = np.empty(self.frame_shape)
        action_frame.fill((action + 1) / self.action_count)
        if len(ob.shape) < len(action_frame.shape):
            ob = np.expand_dims(ob, axis=-1)
        return np.concatenate((ob, action_frame), axis=-1)
    
    def reset(self):
        ob = self.env.reset()
        ob = self._add_action(ob, self.action_space.sample())
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()
    
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        ob = self._add_action(ob, action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info
    
    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=-1)
    
class FrameStackWithAction2D(FrameStackWithAction):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.action_count = env.action_space.n
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shp = shp + (1,) if len(shp) == 2 else shp
        self.frame_shape = (shp[0], shp[1], 1)
        low = env.observation_space.low
        high = env.observation_space.high
        if not np.isscalar(low):
            low = np.min(low)
        if not np.isscalar(high):
            high = np.max(high)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(shp[0], shp[1], (shp[2] + 1) * k),
            dtype=env.observation_space.dtype)

class FrameStackWithAction1D(FrameStackWithAction):
    def __init__(self, env, k, low=0, high=255):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.action_count = env.action_space.n
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        shp = shp + (1,) if len(shp) == 1 else shp
        self.frame_shape = (shp[0], 1)
        low = env.observation_space.low
        high = env.observation_space.high
        if not np.isscalar(low):
            low = np.min(low)
        if not np.isscalar(high):
            high = np.max(high)
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(shp[0], (shp[1] + 1) * k),
            dtype=env.observation_space.dtype)
    
def wrap_muzero(env, dim=96, framestack=32):
    # This wrapper will be added in the RolloutWorker constructor.
    # env = MonitorEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    if 'NoFrameskip' in env.spec.id:
        env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ResizeFrame(env, dim)
    env = FrameStackWithAction2D(env, framestack)
    return env

def wrap_cartpole(env, framestack=16):
    env = FrameStackWithAction1D(env, framestack, low=env.observation_space.low)
    return env

def register_atari_env(env_name: str, muzero_env_name: str, framestack=32):
    tune.register_env(muzero_env_name, lambda ctx: wrap_muzero(gym.make(env_name), framestack=framestack))

def register_cartpole_env(env_name: str, muzero_env_name: str, framestack=16):
    tune.register_env(muzero_env_name, lambda ctx: wrap_cartpole(gym.make(env_name), framestack=framestack))
