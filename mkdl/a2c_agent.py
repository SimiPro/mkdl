import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines import bench
from baselines.a2c.a2c import learn
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

from mario_env import MarioEnv


def train(num_timesteps, seed, policy, lrschedule, num_cpu):
    # TODO: Just fuckin ugly handle that better
    def make_env(rank):
        def _thunk():
            print(rank)
            if num_cpu == 0:
                env = MarioEnv(num_steering_dir=11)
            else:
                env = MarioEnv(num_steering_dir=11, num_env=rank)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            return env
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    learn(policy_fn, env, seed, nsteps=128, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    args = parser.parse_args()
    logger.configure()
    train(num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_cpu=8)


if __name__ == '__main__':
    main()
