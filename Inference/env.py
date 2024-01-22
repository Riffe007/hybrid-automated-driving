import numpy as np
import gym
from PIL import Image
from gym.envs.box2d.car_racing import CarRacing
from gym.spaces import Box

# Constants
SCREEN_X = 64
SCREEN_Y = 64

def process_frame(frame):
    """
    Resize the frame to 64x64 dimensions and normalize pixel values.
    """
    obs = frame[0:84, :, :].astype(np.float32) / 255.0
    obs = Image.fromarray(obs)
    obs = obs.resize((SCREEN_X, SCREEN_Y), Image.ANTIALIAS)
    return np.array(obs, dtype=np.uint8)

class CarRacingWrapper(CarRacing):
    """
    Custom CarRacing environment with modified observation space and step function.
    """
    def __init__(self, full_episode=False):
        super().__init__()
        self.full_episode = full_episode
        self.observation_space = Box(low=0, high=255, shape=(SCREEN_X, SCREEN_Y, 3), dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return process_frame(obs), reward, False if self.full_episode else done, info

def make_env(env_name, seed=-1, render_mode=False, full_episode=False):
    """
    Create and return the customized CarRacing environment.
    """
    env = CarRacingWrapper(full_episode=full_episode)
    if seed >= 0:
        env.seed(seed)
    return env

def game_runner():
    """
    Game runner for human interaction with the CarRacing environment.
    """
    from pyglet.window import key
    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, mod):
        if k == key.LEFT:  a[0] = -1.0
        if k == key.RIGHT: a[0] = +1.0
        if k == key.UP:    a[1] = +1.0
        if k == key.DOWN:  a[2] = +0.8

    def key_release(k, mod):
        if k in [key.LEFT, key.RIGHT] and a[0] != 0: a[0] = 0
        if k == key.UP:    a[1] = 0
        if k == key.DOWN:  a[2] = 0

    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release

    while True:
        env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps == 900 or done:
                break
            steps += 1
            env.render()
        print(f"Human Intelligence Result: Total Steps: {steps}, Total Reward: {total_reward:.0f}")

if __name__ == "__main__":
    game_runner()
