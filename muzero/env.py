from collections import deque
import gym
import numpy as np
from ray.rllib.env.atari_wrappers import (
    MonitorEnv, NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv,
    WarpFrame, FrameStack
)
import ray.tune as tune

import cv2
cv2.ocl.setUseOpenCL(False)


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
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.action_count = env.action_space.n
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.frame_shape = (shp[0], shp[1], 1)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(shp[0], shp[1], (shp[2] + 1) * k),
            dtype=env.observation_space.dtype)
    
    def _add_action(self, ob, action):
        """Add frame with constant value representing action. Notice the plus one."""
        action_frame = np.empty(self.frame_shape)
        action_frame.fill((action + 1) / self.action_count)
        return np.concatenate((ob, action_frame), axis=2)
    
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
        return np.concatenate(self.frames, axis=2)
    
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
    env = FrameStackWithAction(env, framestack)
    return env

def register_muzero_env(env_name: str, muzero_env_name: str):
    tune.register_env(muzero_env_name, lambda ctx: wrap_muzero(gym.make(env_name)))
