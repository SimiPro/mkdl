# Mario Kart 64 Deep Learning Bot

We try to develop our own Mario Kart bot.

## How is our code structured 

### big picture

#### interaction with bizhawk
The python code creates multiple mario environments in its code. This code below creates 1 Environment. 
```python
env = MarioEnv(num_steering_dir=11, jump=True)
```
Based on the gym environment from https://github.com/openai/gym. Each mario environment starts its own python server. Then we start via some environment variables who point to the bizhawk installation, the bizhawk client. For each environment 1 client which then reaches out to the started python servers and connects on it. So we have an 1 to 1 sockets between mario environments on the python side and the mario clients on the bizhawk side. Via this socket we have a primitive text based communication with a few commands e.g. "RESET, 0.1234:1, ..". 

The whole communication is basically encapsulated into the class MarioConnection and the agent only interacts with the MarioEnvironment. The MarioEnv implements the proposed Gym methods from openai: 
- Reset -> State: Resets the environment and gets the initial status back. 
- Act(action) -> State,Reward,Done: Executes an action and gets the reward back and if we're done or not. 
There would be some more methods possibly to use but we don't need them. 
In our case the state is the current screenshot. The reward is some number in [0,1).

MarioConnection encapsulates the byte communication between bizhawk and python.

#### possible simple random agent
To get startet you can check out the simple [random action agent](https://github.com/SimiPro/mkdl/tree/master/mkdl/ppo2_agent.py) which executes random actions. 

#### lua concerning side nodes

Since bizhawk can't currently accept some number to indicate which python server to connect. We had to create a script for each specific port to connect. This is the reason for the numerous new_mario_env0.lua, new_mario_env1.lua etc. 
Since everything starts automatically. 


#### bizhawk side notes
When bizhwak is loaded it loads the state saved on state 2. http://tasvideos.org/Bizhawk/SavestateFormat.html 
You can save a state via: shift + F2 on 2. 
Do this on the start of the track so the agent can directly start to cruse. 

#### mkdl 
mkdl holds all the python code. 
Short summary: 
* [start_bizhawk.py](https://github.com/SimiPro/mkdl/tree/master/mkdl/start_bizhawk.py) : responsible for starting bizhawk automatically
* [utils.py](https://github.com/SimiPro/mkdl/tree/master/mkdl/utils.py): some util functions
* [run_bizhawk.py](https://github.com/SimiPro/mkdl/tree/master/run_bizhawk.py): starts bizhawk from python console
* [policy.py](https://github.com/SimiPro/mkdl/tree/master/mkdl/policy.py): holds our used NeuralNetworks. Mostly used is OurCNN2 which is a CNN with 2 additional non-linear relu layers. 
* [mario_env.py](https://github.com/SimiPro/mkdl/tree/master/mkdl/mario_env.py): holds the MarioEnv which the agent can act on. And also the MarioConnection which communicates with the Bizhawk
* [ppo2_agent.py](https://github.com/SimiPro/mkdl/tree/master/mkdl/ppo2_agent.py): executes the ppo agent. [openai ppo implementation](https://github.com/openai/baselines/tree/master/baselines/ppo2)
* [a2c_agent.py](https://github.com/SimiPro/mkdl/tree/master/mkdl/a2c_agent.py)
* [a3c_agent.py](https://github.com/SimiPro/mkdl/tree/master/mkdl/a3c_agent.py)
* just some additional agents we tried out and rejected again for our problem. 

### Prerequisites

* windows only (thanks to bizhawk)
* python 3.6
* a few additional libraries to get the gym environment up and running. sorry for that

## Contributing

* Huy Cao Tri Do
* Simon Huber

## License

This project is licensed under the MIT License

## Acknowledgments

* big thanks to https://github.com/rameshvarun/NeuralKart which was basically 
our inspiration for doing this. 

