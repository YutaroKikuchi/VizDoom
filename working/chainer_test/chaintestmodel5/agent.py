# from https://github.com/icoxfog417/chainer_pong/blob/master/model/agent.py


class Agent():
    # Father of DQN agent and DQN trainer : useful to have only one loop for both training and testing in environment

    def __init__(self, actions):
        self.actions = actions

    def start(self, observation):
        return 0  # default action

    def act(self, observation, reward):
        return 0  # default action
    
    def end(self, observation, reward):
        pass

    def report(self, episode):
        return ""