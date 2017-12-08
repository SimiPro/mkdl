import queue
import threading

import gym
import socket
import logging
import numpy as np
import time

from PIL import ImageGrab, Image
from gym import spaces

from start_bizhawk import start_mario
from utils import Singleton

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_WIDTH = 200
INPUT_HEIGHT = 66
INPUT_CHANNELS = 3


class MarioEnv(gym.Env):
    """
    Mario Environment. Use this to communicate with the Mario kart client
    """
    def __init__(self,  num_steering_dir=0, num_env=-1):
        """if you have multiple threads input marioserverholder"""
        # Set these in ALL subclasses
        self.mario_server = MarioServer(num_env=num_env)
        self.mario_server.start()
        self.action_space = spaces.Discrete(num_steering_dir)
        self.observation_space = spaces.Box(low=0, high=255, shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
        self.mario_connection = MarioConnection(self.mario_server, num_env=num_env)
        self.num_steering_dir = num_steering_dir

    # we send here an action to execute to the mario game
    # we expect a (new screenshot_path, reward, done) response
    def _step(self, action):
        if self.num_steering_dir > 0:  # we use action encoding
            action_space = np.linspace(-1, 1, self.num_steering_dir)
            action = action_space[action]

        logger.debug('executing action: {}'.format(action))
        (screen_shot, reward, done) = self.mario_connection.send_action(action)

        im = self.get_screenshot(screen_shot)
        im = self.prepare_image(im)

        return im, reward, done, {}

    def _reset(self):
        (screen_shot_path, reward, done) = self.mario_connection.reset_client()

        im = self.get_screenshot(screen_shot_path)
        im = self.prepare_image(im)

        return im

    def _render(self, mode='human', close=False):
        # we always render human at the moment
        return

    def _seed(self, seed=None):
        return []

    def prepare_image(self, im):
        return self.prepare_image_(im, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)

    def prepare_image_(self, im, conv_input_width, conv_input_height, conv_input_channels):
        im = im.resize((conv_input_width, conv_input_height))
        im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((conv_input_height, conv_input_width, conv_input_channels))
        #im_arr = np.expand_dims(im_arr, axis=0)
        return im_arr

    def get_screenshot(self, screenshot_path):
        im = None
        i = 0
        while im is None and i < 5:
            i = i + 1
            if screenshot_path == 'clip':
                im = ImageGrab.grabclipboard()
            else:
                im = Image.open(screenshot_path)

        if im is not None:
            self.old_image = im
            return im
        print('had to take old image!! failed to take screenshot!')
        return self.old_image


class MarioServer(threading.Thread):
    def __init__(self, port=36295, num_env=-1):
        super().__init__()
        #print("port: {} | num_env: {}".format(port, num_env))
        if num_env is not (-1):
            self.port = port + num_env
        else:
            self.port = port
        print("port: {} | num_env: {}".format(self.port, num_env))
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        name = 'localhost'
        self.server_socket.bind((name, self.port))
        self.connections = 0
        self.connection_queue = queue.Queue()
        print('mario server | ip: {} | port: {} - connect to it'.format(name, self.port))

    def run(self):
        logger.info('starting tcp - socket on port: {}'.format(self.port))
        self.server_socket.listen(1)  # wait for 5 connections
        while self.connections < 1:
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


class MarioConnection:
    """
    Responsible for the socket communication between the python server
    and the lua socket of the client
    """
    def __init__(self, server, num_env=-1):
        start_mario(num_env=num_env)  # start mario then get connection. server should be up and running already
        (self.client_socket, self.client_address) = server.get_connection_blocking()

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
