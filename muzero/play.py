import base64
import gym
import io    
import numpy as np
import PIL.Image
import pyglet
import time
from xvfbwrapper import Xvfb

def jupyter_play(env, agent, steps=1000):
  from IPython import display

  policy = agent.get_policy()
  
  vdisplay = Xvfb(width=1280, height=740)
  vdisplay.start()

  observation = env.reset()

  def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = io.BytesIO()
    ima = PIL.Image.fromarray(a).save(f, fmt)
    return f.getvalue()

  imagehandle = display.display(display.Image(data=showarray(env.render(mode='rgb_array')), width=450), display_id='gymscr')

  action = 0
  reward = 0
  for _ in range(steps):
    time.sleep(0.001)
    action = agent.compute_action(observation, prev_reward=reward, prev_action=action)
    observation, reward, done, info = env.step(action)
    
    display.update_display(display.Image(data=showarray(env.render(mode='rgb_array')), width=450), display_id='gymscr')
    if done: break

  vdisplay.stop()

def mp4_play(env, agent, steps=1000):
  policy = agent.get_policy()

  env = gym.wrappers.Monitor(env, "./gym-results", force=True, video_callable=lambda x: True)    
  observation = env.reset()

  action = 0
  reward = 0
  for i in range(steps):
    env.render()
    action = agent.compute_action(observation, prev_reward=reward, prev_action=action)
    observation, reward, done, info = env.step(action)
      
    if done:
      #print(f'done in {i} steps')
      break
  else:
    # Manually save if the environment was not done.
    env.stats_recorder.save_complete()
    env.stats_recorder.done = True

  env.close()
  return f'./gym-results/openaigym.video.{env.file_infix}.video000000.mp4'

def jupyter_mp4_replay(path):
  from IPython.display import HTML

  video = io.open(path, 'r+b').read()
  encoded = base64.b64encode(video)
  HTML(data='''
      <video width="360" height="auto" alt="test" controls><source src="data:video/mp4;base64,{0}" type="video/mp4" /></video>
  '''.format(encoded.decode('ascii')))
