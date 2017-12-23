import sys
import argparse
from baselines import bench, logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
import gym
import logging
import multiprocessing
import os.path as osp
import tensorflow as tf

from mario_env import MarioEnv
from policy import ContCnnPolicy, OurCNN

def run(env_id, num_timesteps, seed, policy):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()
    nenvs = 1
    def make_env(rank):
        def env_fn():
            print(rank)
            if nenvs == 1:
                env = MarioEnv(num_steering_dir=11)
            else:
                env = MarioEnv(num_steering_dir=11, num_env=rank)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            return env
        return env_fn

    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, 4)
    policy = {'cont': ContCnnPolicy, 'cnn' : OurCNN, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy}[policy]
    ppo2.run_only(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 1e-3,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
               save_interval=10)


def train(env_id, num_timesteps, seed, policy):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    gym.logger.setLevel(logging.WARN)
    tf.Session(config=config).__enter__()
    nenvs = 8
    def make_env(rank):
        def env_fn():
            print(rank)
            if nenvs == 1:
                env = MarioEnv(num_steering_dir=11)
            else:
                env = MarioEnv(num_steering_dir=11, num_env=rank)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and osp.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            return env
        return env_fn

    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    set_global_seeds(seed)
    env = VecFrameStack(env, 4)
    policy = {'cont': ContCnnPolicy, 'cnn' : OurCNN, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy}[policy]
    ppo2.learn(policy=policy, env=env, nsteps=128, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 1e-3,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
               save_interval=10)

    # noptepochs how many times learning from a batch
    # nminibatches how many samples should we learn from. bigger if cont. space
    # ent_coef = entropy how much caos=?
    # lr = learning rate

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, policy=args.policy)

if __name__ == '__main__':
    main()



