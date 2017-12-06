import logging
import os
from train import create_model
import reinforcement
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OldAgent(object):
    """
    This class implements the reference agent
    ported old supervised agent to the reinforcement model
    """
    def __init__(self):
        self.model = create_model(keep_prob=1)
        if os.path.isfile('weights/all.hdf5'):
            logger.info('loading weights from weights/all.hdf5')
            self.model.load_weights('weights/all.hdf5')

    def act(self, screenshot_path, reward, done):
        action = None
        im = utils.get_screenshot(screenshot_path)
        if im is not None:
            prepared_image = utils.prepare_image(im)
            action = self.model.predict(prepared_image, batch_size=1)[0]
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
