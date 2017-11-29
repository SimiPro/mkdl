import sys, time, logging, os, argparse
import numpy as np
from PIL import Image, ImageGrab
from train import create_model, is_valid_track_code, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS
import reinforcement

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_image(im):
    im = im.resize((INPUT_WIDTH, INPUT_HEIGHT))
    im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS))
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr


class OldAgent(object):
    """ ported old supervised agent to the reinforcement model"""
    def __init__(self):
        self.model = create_model(keep_prob=1)
        if os.path.isfile('weights/all.hdf5'):
            logger.info('loading weights from weights/all.hdf5')
            self.model.load_weights('weights/all.hdf5')

    def act(self, screenshot_path, reward, done):
        action = None
        if screenshot_path == 'clip':
            im = ImageGrab.grabclipboard()
            if im is not None:
                action = self.model.predict(prepare_image(im), batch_size=1)[0]
                action = action[0]
            else:
                logger.error('If you want image from clipboard, provide image in clipboard')
        if action is None:
            logger.error('could not predict next action set action=0 | screenshot path: {}'.format(screenshot_path))
            return 0
        return action


if __name__ == '__main__':
    mario_env = reinforcement.MarioEnv()
    agent = OldAgent()

    (screenshot_path, reward, done) = mario_env.reset()

    while not done:
        action = agent.act(screenshot_path, reward, done)
        (screenshot_path, reward, done) = mario_env.step(action)
