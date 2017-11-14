# example from https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb modified for doom
from __future__ import print_function
from vizdoom import *

from random import choice
from time import sleep

import datetime
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
from chainerrl import env
from chainerrl import spaces
import gym
import numpy as np

from skimage.color import rgb2gray
from skimage.transform import resize

n_episodes = 220
save_every = 5


class DOOM_ENV(env.Env):
    """Arcade Learning Environment."""

    def __init__(self, scenario="./config/basic.wad", dmap = "map01", episode_len=200, window=True):
        self.game = DoomGame()
        self.game.set_doom_scenario_path(scenario)
        self.game.set_doom_map(dmap)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_screen_format(ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_automap_buffer_enabled(True)
        self.game.set_render_hud(False)
        self.game.set_render_minimal_hud(False)  # If hud is enabled
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)  # Bullet holes and blood on the walls
        self.game.set_render_particles(False)
        self.game.set_render_effects_sprites(False)  # Smoke and blood
        self.game.set_render_messages(False)  # In-game messages
        self.game.set_render_corpses(False)
        self.game.set_render_screen_flashes(True) # Effect upon taking damage or picking up items
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.ATTACK)
        self.game.add_available_game_variable(GameVariable.AMMO2)
        self.game.set_episode_timeout(episode_len)
        self.game.set_episode_start_time(10)
        self.game.set_window_visible(window)
        self.game.set_sound_enabled(True)
        self.game.set_living_reward(-1)
        self.game.set_mode(Mode.PLAYER)
        self.game.init()
        self.game.new_episode()
        self._reward=0
        self.frame = 0
        self.legal_actions = [[True, False, False], [False, True, False], [False, False, True]]
        self.episode_len = episode_len
        obs = resize(rgb2gray(self.game.get_state().screen_buffer), (80, 80))
        obs = obs[np.newaxis, :, :]
        self.current_screen = obs

    @property
    def state(self):
        if self.game.is_episode_finished():
            return self.current_screen
        rr = self.game.get_state()
        obs = resize(rgb2gray(rr.screen_buffer), (80, 80))
        obs = obs[np.newaxis, :, :]
        self.current_screen = obs
        return obs
    
    def random_action_doom(self,nothing=0) :
        result = choice(range(0,len(self.legal_actions)))
        return result

    @property
    def is_terminal(self):
        if self.game.is_episode_finished():
            return True
        return self.frame == self.episode_len-1

    @property
    def reward(self):
        return self._reward

    @property
    def number_of_actions(self):
        return len(self.legal_actions)

    def receive_action(self, action):
        self._reward = self.game.make_action(self.legal_actions[action])
        return self._reward

    def initialize(self):
        self.game.new_episode()
        self._reward = 0
        self.frame = 0

    def reset(self):
        self.initialize()
        return self.state

    def step(self, action):
        self.frame = self.frame + 1
        self.receive_action(action)
        return self.state, self.reward, self.is_terminal, {}

    def close(self):
        pass


#---------------------------------------------------------------------------
env = DOOM_ENV()
obs = env.reset()

class QFunction(chainer.Chain):
    def __init__(self, n_history=1, n_action=6):
        super().__init__(
            l1=L.Convolution2D(n_history, 32, ksize=8, stride=4, nobias=False),
            l2=L.Convolution2D(32, 64, ksize=3, stride=2, nobias=False),
            l3=L.Convolution2D(64, 64, ksize=3, stride=1, nobias=False),
            l4=L.Linear(3136, 512),
            out=L.Linear(512, n_action, initialW=np.zeros((n_action, 512), dtype=np.float32))
        )

    def __call__(self, x, test=False):
        s = chainer.Variable(x)
        h1 = F.relu(self.l1(s))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = self.out(h4)
        return chainerrl.action_value.DiscreteActionValue(h5)

n_action = env.number_of_actions
print("n action size obs")
print(n_action)
n_history=1
q_func = QFunction(n_history, n_action)

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

gamma = 0.95

explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.random_action_doom)

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 4)

phi = lambda x: x.astype(np.float32, copy=False)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    minibatch_size=4, replay_start_size=500,
     phi=phi)

last_time = datetime.datetime.now()

filename = "toreplace"

chainerrl.experiments.train_agent_with_evaluation(
    agent, env,
    steps=2000,           # Train the agent for 2000 steps
    eval_n_runs=10,       # 10 episodes are sampled for each evaluation
    max_episode_len=200,  # Maximum length of each episodes
    eval_interval=1000,   # Evaluate the agent after every 1000 steps
    outdir='result')      # Save everything to 'result' directory

print("demo starts")
#agent.load(filename)

for i in range(1, n_episodes + 1):
    obs = resize(rgb2gray(env.reset()),(80,80))
    obs = obs[np.newaxis, :, :]

    reward = 0
    done = False
    R = 0

    while not done:
        action = agent.act(obs)
        #action = agent.act_and_train(obs, reward)
        #action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        obs = resize(rgb2gray(obs), (80, 80))
        obs = obs[np.newaxis, :, :]


    agent.stop_episode()
print("demo ended")

"""
env = DOOM_ENV()
obs = env.reset()

class QFunction(chainer.Chain):
    def __init__(self, n_history=1, n_action=6):
        super().__init__(
            l1=L.Convolution2D(n_history, 32, ksize=8, stride=4, nobias=False),
            l2=L.Convolution2D(32, 64, ksize=3, stride=2, nobias=False),
            l3=L.Convolution2D(64, 64, ksize=3, stride=1, nobias=False),
            l4=L.Linear(3136, 512),
            out=L.Linear(512, n_action, initialW=np.zeros((n_action, 512), dtype=np.float32))
        )

    def __call__(self, x, test=False):
        s = chainer.Variable(x)
        h1 = F.relu(self.l1(s))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = self.out(h4)
        return chainerrl.action_value.DiscreteActionValue(h5)

n_action = env.number_of_actions
print("n action size obs")
print(n_action)
n_history=1
q_func = QFunction(n_history, n_action)

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

gamma = 0.95

explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.random_action_doom)

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 4)

phi = lambda x: x.astype(np.float32, copy=False)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    minibatch_size=4, replay_start_size=500,
     phi=phi)

last_time = datetime.datetime.now()

filename = "toreplace"
for i in range(1, n_episodes + 1):
    obs = resize(rgb2gray(env.reset()),(80,80))
    obs = obs[np.newaxis, :, :]

    reward = 0
    done = False
    R = 0

    while not done:
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        obs = resize(rgb2gray(obs), (80, 80))
        obs = obs[np.newaxis, :, :]

        if reward != 0:
            R += reward

    elapsed_time = datetime.datetime.now() - last_time
    print('episode:', i, '/', n_episodes,
          'reward:', R,
          'minutes:', elapsed_time.seconds/60)
    last_time = datetime.datetime.now()

    if i % save_every == 0:
        filename = 'agent_Breakout' + str(i)
        agent.save(filename)

    agent.stop_episode_and_train(obs, reward, done)
print('Finished. Now testing')

print("demo starts")
#agent.load(filename)

for i in range(1, n_episodes + 1):
    obs = resize(rgb2gray(env.reset()),(80,80))
    obs = obs[np.newaxis, :, :]

    reward = 0
    done = False
    R = 0

    while not done:
        action = agent.act(obs)
        #action = agent.act_and_train(obs, reward)
        #action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        obs = resize(rgb2gray(obs), (80, 80))
        obs = obs[np.newaxis, :, :]


    agent.stop_episode()
print("demo ended")
"""