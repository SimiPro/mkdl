import time
import utils
import logging
import random
import sys
import numpy as np
import threading
import tensorflow as tf
from PIL import ImageGrab, Image
from keras.models import *
from keras.layers import *
from keras import layers, models
from keras import backend as K
from reinforcement import MarioEnv

THREAD_DELAY = .1
N_STEP_RETURN = 8  # learn after 5 examples (cus then we know the future)

NUM_ACTIONS = 50
MIN_BATCH = 32  # 32
LEARNING_RATE = 5e-3


EPS_START = 0.001
EPS_STOP  = .001
EPS_STEPS = 75000

GAMMA = 0.99
GAMMA_N = GAMMA ** N_STEP_RETURN

LOSS_V = .5			# v loss coefficient
LOSS_ENTROPY = .001 	# entropy coefficient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


INPUT_WIDTH = 200
INPUT_HEIGHT = 66
INPUT_CHANNELS = 3


class EnvironmentRunner(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)
        self.stop_signal = False
        # TODO: we should make mario here ourself but we can't at the momment
        # because the fuckin bizhawk doesn't start from console with all ready at the moment
        # fuckin buillshit
        self.old_image = None
        self.agent = Agent()

    def run_episode(self):
        (s, r, d) = self.mario_env.reset()
        counter = 0
        while True:
            time.sleep(THREAD_DELAY)  # so that multiple agents can run in parallel (more then cpus)
            try:
                im = self.get_screenshot(s)
                im = self.prepare_image(im)
                a = self.agent.act(im)
                s_, r, done = self.mario_env.step(a)
                im_ = utils.get_screenshot(s_)
                im_ = utils.prepare_image(im_)
                if not done:
                    self.agent.train(im, a, r, im_)
                s = s_
            except:
                print('some error occured during image processing use default action')
                print(sys.exc_info()[0])
                a = 0

            if counter == 300:
                (s, r, d) = self.mario_env.reset()
                counter = 0
                global frames
                frames = 0

            counter += 1

            if done or self.stop_signal:
                break

    def prepare_image(self, im):
        return self.prepare_image_(im, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)

    def prepare_image_(self, im, conv_input_width, conv_input_height, conv_input_channels):
        im = im.resize((conv_input_width, conv_input_height))
        im_arr = np.frombuffer(im.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((conv_input_height, conv_input_width, conv_input_channels))
        im_arr = np.expand_dims(im_arr, axis=0)
        return im_arr

    def get_screenshot(self, screenshot_path):
        im = None
        if screenshot_path == 'clip':
            im = ImageGrab.grabclipboard()
        else:
            im = Image.open(screenshot_path)
        if im is not None:
            self.old_image = im
            return im
        print('had to take old image!! failed to take screenshot!')
        return self.old_image

    def run(self):
        self.mario_env = MarioEnv(num_steering_dir=NUM_ACTIONS)  # this should run in async context
        while not self.stop_signal:
            self.run_episode()

    def stop(self):
        self.stop_signal = True

frames = 0
class Agent:
    def __init__(self):
        self.memory = []
        self.eps_start = EPS_START
        self.eps_end = EPS_STOP
        self.eps_steps = EPS_STEPS

    def act(self, image):
        # simple epsilon greedy policy
        # implement linear decreasing epsilon
        global frames
        frames = frames + 1
        if random.random() < self.get_epsilon():
            return random.randint(0, NUM_ACTIONS-1)
        else:
            p = brain.predict_p(image)
            return np.argmax(p)
            #return np.random.choice(NUM_ACTIONS, p=p)


    def train(self, s, a, r, s_):
        actions = np.zeros(NUM_ACTIONS)
        actions[a] = 1
        self.memory.append((s, actions, r, s_))

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)
            self.memory.pop(0)

        # TODO: What if s_ is null (terminal state?)

    def get_epsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate


def get_sample(memory, n):
    r = 0.
    for i in range(n):
        logger.debug("memory[i][2] = reward = {}".format(memory[i][2]))
        r += float(memory[i][2]) * (GAMMA ** i)  # discount future reward
    s, a, _, _ = memory[0]  # first image
    _, _, _, s_ = memory[n-1]  # last image
    return s, a, r, s_


class Brain:
    """  Encapsulates our neural network
    training_queue: 5 arrays:
        - starting state s
        - one-hot encoded taken action a
        - discounted n-step return r
        - end state s_
        - done or not_
    """
    train_queue = [[], [], [], [], []]
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        # see this
        K.set_learning_phase(1)  # set learning phase

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        if os.path.isfile('weights/weights_simi.h5'):
            print('load weights from file')
            self.model.load_weights('weights/weights_simi.h5')

        self.default_graph.finalize()  # supress modifications



    def _build_model(self):
        input_ = layers.Input(batch_shape=(None, utils.INPUT_HEIGHT, utils.INPUT_WIDTH, utils.INPUT_CHANNELS))

        #norm1 = layers.BatchNormalization()(input_)
        conv1 = layers.Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu')(input_)

        #norm2 = layers.BatchNormalization()(conv1)
        conv2 = layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')(conv1)

        #norm3 = layers.BatchNormalization()(conv2)
        conv3 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv2)

        #norm4 = layers.BatchNormalization()(conv3)
        #conv4 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(norm4)

        flatten = layers.Flatten()(conv3)
        nonlin1 = layers.Dense(512, activation='relu')(flatten)
        #dropout1 = layers.Dropout(0.5)(nonlin1)

        #nonlin2 = layers.Dense(100, activation='relu')(dropout1)
        #dropout2 = layers.Dropout(0.5)(nonlin2)

        #nonlin3 = layers.Dense(50, activation='relu')(dropout2)
        #dropout3 = layers.Dropout(0.5)(nonlin3)

        #nonlin4 = layers.Dense(10, activation='relu')(dropout3)

        out_actions = layers.Dense(NUM_ACTIONS, activation='softmax')(nonlin1)
        out_value = layers.Dense(1, activation='linear')(nonlin1)

        model = models.Model(inputs=[input_], outputs=[out_actions, out_value])
        model._make_predict_function()  # initialize

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, utils.INPUT_HEIGHT, utils.INPUT_WIDTH, utils.INPUT_CHANNELS))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1, keep_dims=True)  # maximize entropy(regu)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                return 	# we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5*MIN_BATCH:
            print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})
        print('saving model')
        self.model.save_weights('weights/weights_simi.h5')

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)
            if s_ is None:
                raise NotImplementedError("implement if s_ is none (when we're done)")
                self.train_queue[3].append(None)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


THREADS = 4
OPTIMIZERS = 2

RUN_TIME = 100000

if __name__ == '__main__':
    print('going for it')
    brain = Brain()
    envs = [EnvironmentRunner() for i in range(THREADS)]
    opts = [Optimizer() for i in range(OPTIMIZERS)]

    for each_optimizer in opts:
        each_optimizer.start()

    for each_env in envs:
        each_env.start()

    time.sleep(RUN_TIME)

    for each_env in envs:
        each_env.stop()
    for each_env in envs:
        each_env.join()

    for each_optimizer in opts:
        each_optimizer.stop()
    for each_optimizer in opts:
        each_optimizer.join()

    print('training finished')
