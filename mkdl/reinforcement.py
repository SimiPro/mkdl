import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import tcp_server
import time
from threading import Thread


class MarioEnv(gym.Env):
    def __init__(self):
        self.in_game = False
        # Set these in ALL subclasses
        self.action_space = None #spaces.Box(low=0, high=1)
        self.observation_space = None

    def game_started(self):
        print("MARIO Environment is started!")
        self.in_game = True

    # Override in ALL subclasses
    def _step(self, action):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _render(self, mode='human', close=False):
        # we always render human at the moment
        return

    def _seed(self, seed=None):
        return []


if __name__ == "__main__":
    tcp_server.mario_env = MarioEnv()
    tcp_server.start_server()
