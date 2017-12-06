import queue
import threading

import gym
import socket
import logging
import numpy as np
import time

from start_bizhawk import start_mario
from utils import Singleton

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarioEnv(gym.Env):
    """
    Mario Environment. Use this to communicate with the Mario kart client
    """
    def __init__(self, num_steering_dir=0):
        self.in_game = False
        # Set these in ALL subclasses
        self.action_space = None  # spaces.Box(low=0, high=1)
        self.observation_space = None
        self.mario_connection = MarioConnection()
        self.num_steering_dir = num_steering_dir

    def game_started(self):
        print("MARIO Environment is started!")
        self.in_game = True

    def action_response(self, screenshot_path, reward, done):
        self.screenshot_path = screenshot_path
        self.reward = reward
        self.done = done

    # we send here an action to execute to the mario game
    # we expect a (new screenshot_path, reward, done) response
    def _step(self, action):
        if self.num_steering_dir > 0:  # we use action encoding
            action_space = np.linspace(-1, 1, self.num_steering_dir)
            action = action_space[action]

        logger.debug('executing action: {}'.format(action))
        (screen_shot, reward, done) = self.mario_connection.send_action(action)
        return (screen_shot, reward, done)

    def _reset(self):
        (screen_shot, reward, done) = self.mario_connection.reset_client()
        return (screen_shot, reward, done)

    def _render(self, mode='human', close=False):
        # we always render human at the moment
        return

    def _seed(self, seed=None):
        return []


@Singleton
class MarioServer(threading.Thread):
    def __init__(self):
        super().__init__()
        self.port = 36296
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', self.port))
        self.connections = 0
        self.connection_queue = queue.Queue()
        print('mario server runs - connect to it')

    def run(self):
        logger.info('starting tcp - socket on port: {}'.format(self.port))
        self.server_socket.listen(5)  # wait for 5 connections
        while self.connections < 5:
            logger.info('Waiting for some more connection')
            conn = self.server_socket.accept()
            self.connection_queue.put(conn)

    def get_connection_blocking(self):
        result = None
        print('wait for client connection')
        while result is None:
            if self.connection_queue.empty():
                time.sleep(0.5)  # yield
            else:
                result = self.connection_queue.get()
        print('got one connection')
        (client_socket, client_address) = result
        return client_socket, client_address


server = MarioServer.Instance()
server.start()


class MarioConnection:
    """
    Responsible for the socket communication between the python server
    and the lua socket of the client
    """
    def __init__(self):
        start_mario()  # start mario then get connection. server should be up and running already
        (self.client_socket, self.client_address) = MarioServer.Instance().get_connection_blocking()

    def reset_client(self):
        self.client_socket.send(b'RESET\n')
        return self.expect_answer()

    def send_action(self, action):
        logger.info('sending action: {}'.format(action))
        self.client_socket.send(bytes("{}\n".format(action), 'utf-8'))
        return self.expect_answer()

    def expect_answer(self):
        """ We expect screenshot_path, reward, done from client """
        message = self.client_socket.recv(1024)
        message = message.decode('utf-8')
        #  logger.info('received data: {}'.format(message))
        if message.startswith("MESSAGE"):
            parsed = message.split("_")
            #  logger.info("splitted message: {}".format(parsed))

            screenshot_path = parsed[1]
            reward = parsed[3]
            done = parsed[5]

            if done.startswith("False"):
                done = False
            else:
                done = True

            logger.info("Message received: screen_shot: {}, reward: {}, done:{}".format(screenshot_path, reward, done))
            return (screenshot_path, reward, done)
        else:
            logger.error("Got some screwed up message: {}".format(message))
            return None


"""
this main program just runs 50 steps forward then it resets
and then it drives right for 50 steps
"""
if __name__ == "__main__":
    mario = MarioEnv()
    (screen_shot, reward, done) = mario.reset()

    for i in range(50):
        (screen_shot, reward, done) = mario.step(0)

    (screen_shot, reward, done) = mario.reset()

    for i in range(50):
        (screen_shot, reward, done) = mario.step(1)
