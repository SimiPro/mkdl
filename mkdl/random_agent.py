from mario_env import MarioEnv

"""

"""
def main():
    env = MarioEnv(num_steering_dir=11, jump=False)
    action_space = env.action_space

    state = env.reset()
    done = False
    while not done:
        new_action = action_space.sample()
        new_state, reward, done, info  = env.step(new_action)
    env.close()

if __name__ == '__main__':
    main()
