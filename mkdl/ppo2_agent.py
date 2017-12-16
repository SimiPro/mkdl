import sys
import argparse
from baselines import bench, logger
from baselines.a2c import utils
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.distributions import make_pdtype
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import gym
import logging
import multiprocessing
import os.path as osp
import tensorflow as tf
import numpy as np

from mario_env import MarioEnv


def train(env_id, num_timesteps, seed, policy):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()
    nenvs = 4
    def make_env(rank):
        def env_fn():
            print(rank)
            if nenvs == 1:
                env = MarioEnv(num_steering_dir=0)
            else:
                env = MarioEnv(num_steering_dir=0, num_env=rank)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            return env
        return env_fn

    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, 4)
    policy = {'cont': ContCnnPolicy, 'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy}[policy]
    ppo2.learn(policy=policy, env=env, nsteps=100, nminibatches=4,
        lam=0.95, gamma=0.95, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
               save_interval=10)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cont')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy)

if __name__ == '__main__':
    main()


class ContCnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.shape[0]
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = utils.conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = utils.conv(h, 'c2', nf=32, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = utils.conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = utils.conv_to_fc(h3)
            h4 = utils.fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            pi = utils.fc(h4, 'pi', nact, act=lambda x:x, init_scale=0.01)
            vf = utils.fc(h4, 'v', 1, act=lambda x:x)[:,0]
            logstd = tf.get_variable(name='logstd', shape=[1, nact], initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi*0.0+logstd], axis=1)
        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value