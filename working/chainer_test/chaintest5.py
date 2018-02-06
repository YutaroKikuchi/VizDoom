# modified https://github.com/icoxfog417/chainer_pong/blob/master/run.py
import os
import sys
a = os.path.dirname(os.path.abspath(__file__)) + "/chaintestmodel5"
sys.path.append(a)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chaintestmodel5.environment import Environment
from chaintestmodel5.dqn_agent import DQNAgent
from chaintestmodel5.dqn_trainer import DQNTrainer

PATH = os.path.join(os.path.dirname(__file__), "./store") # where we store the training data that will be used in the testing


def run(render, gpu, show_end_graph):
    # -- run the test using the training data in the PATH
    # render : True to open the Doom window
    # gpu : True to use the gpu functions (faster)
    
    env = Environment(episode_len=200,show_frames=False, epoch_len = 2000, render=render, show_end_graph=show_end_graph) # chose the len of epoch and episode
    # (show_frames set to True to see the first and last frame of each epoch with the 3 convolutions)
    
    agent = DQNAgent(env.actions, epsilon=0.01, model_path=PATH, on_gpu=gpu) # will load the training data
    
    # play
    for ep, s, r in env.play(agent, epochs=20, render=render, action_interval=4): # chose the number of epochs and action interval
        pass


def train(render, gpu, show_end_graph):
    # -- train an agent and save it to the PATH
    # render : True to open the Doom window
    # gpu : True to use the gpu functions (faster)
    
    env = Environment(episode_len=200,show_frames=False, epoch_len = 2000, render=render, show_end_graph=show_end_graph) # chose the len of epoch and episode
    # (show_frames set to True to see the first and last frame of each epoch with the 3 convolutions)
    
    agent = DQNAgent(env.actions, epsilon=0.5, model_path=PATH, on_gpu=gpu) # the agent making the decisions, gets better with time
    trainer = DQNTrainer(agent) # the one organizing the training session for the agent, same type as agent to re-use the same environment loop in training and testing

    # train
    for ep, s, r in env.play(trainer, epochs=10, render=render, action_interval=4): # chose the number of epochs and action interval
        pass


# Run a training and testing session
train(False, False, True)
run(True, False, True)

