# modified https://github.com/icoxfog417/chainer_pong/blob/master/run.py
import os
import sys
#import gym
a = os.path.dirname(os.path.abspath(__file__)) + "/chaintestmodel5"
print(a)
sys.path.append(a)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from chaintestmodel5.environment import Environment
from chaintestmodel5.dqn_agent import DQNAgent
from chaintestmodel5.dqn_trainer import DQNTrainer


PATH = os.path.join(os.path.dirname(__file__), "./store")
# by default epoch len is 200f

def run(submit_key, gpu):
    env = Environment()
    agent = DQNAgent(env.actions, epsilon=0.01, model_path=PATH, on_gpu=gpu)
    path = ""
    episode = 5
    if submit_key:
        print("make directory to submit result")
        path = os.path.join(os.path.dirname(__file__), "submit")
    epochsnb = 20

    for ep, s, r in env.play(agent, epochs=epochsnb, render=True, action_interval=4, record_path=path):
        pass
    
    #if submit_key:
        #gym.upload(path, api_key=submit_key)


def train(render, gpu):
    env = Environment(show_frames=False) # show_frames set to True to see the first and last frame of each epoch with the 3 convolutions
    agent = DQNAgent(env.actions, epsilon=0.5, model_path=PATH, on_gpu=gpu)
    trainer = DQNTrainer(agent)

    for ep, s, r in env.play(trainer, epochs=20, render=render, report_interval=20, action_interval=4):
        pass

train(True, False)
run(False, False)