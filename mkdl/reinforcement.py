import gym
import socket
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarioEnv(gym.Env):
    def __init__(self):
        self.in_game = False
        # Set these in ALL subclasses
        self.action_space = None  # spaces.Box(low=0, high=1)
        self.observation_space = None
        self.mario_connection = MarioConnection()

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


class MarioConnection:
    def __init__(self, port=36296):
        self.port = port
        self.client_socket = None
        self.client_address = None
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('localhost', self.port))

    def reset_client(self):
        logger.info('starting tcp - socket on port: {}'.format(self.port))
        self.server_socket.listen(1)  # max connections 1 currently
        while self.client_socket is None:
            logger.info("Waiting for game to connect")
            (self.client_socket, self.client_address) = self.server_socket.accept()
        logger.info("Connected by:{}".format(self.client_address))
        self.client_socket.send(b'RESET\n')
        return self.expect_answer()

    def send_action(self, action):
        self.client_socket.send(bytes("{}\n".format(action), 'utf-8'))
        return self.expect_answer()

    def expect_answer(self):
        """ We expect screenshot_path, reward, done from client """
        message = self.client_socket.recv(1024)
        message = message.decode('utf-8')
        logger.info('received data: {}'.format(message))
        if message.startswith("MESSAGE"):
            parsed = message.split(":")
            logger.info("splitted message: {}".format(parsed))

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


if __name__ == "__main__":
    mario = MarioEnv()
    (screen_shot, reward, done) = mario.reset()

    for i in range(50):
        (screen_shot, reward, done) = mario.step(0)

    (screen_shot, reward, done) = mario.reset()

    for i in range(50):
        (screen_shot, reward, done) = mario.step(1)
