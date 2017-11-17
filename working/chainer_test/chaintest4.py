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
import math

from tqdm import trange

from skimage.color import rgb2gray
from skimage.transform import resize

n_epochs = 4
epoch_len = 500
test_epoch_len =8
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
        
        self.action_space = spaces.Discrete(len(self.legal_actions))
        
        one_screen_observation_space = spaces.Box(
            low=0, high=255, shape=(80, 80))
        n_last_screens = 4
        self.observation_space = spaces.Tuple(
            [one_screen_observation_space] * n_last_screens)
        
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
    
    def set_window(self, on):
        self.game.close()
        self.game.set_window_visible(on)
        self.game.init()
        self.game.new_episode()
        return on
    
    def get_total_score(self):
        return self.game.get_total_reward()

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
    def __init__(self, n_history=1, n_action=3):
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

optimizer = chainer.optimizers.Adam(eps=1e-2)#1e-2
optimizer.setup(q_func)

gamma = 0.95

explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.random_action_doom)

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 11)

phi = lambda x: x.astype(np.float32, copy=False)

agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    minibatch_size=4, n_times_update=1, replay_start_size=50, target_update_interval=100,
     phi=phi)


env.set_window(True)


for i in range(1, 1 + 1):
    obs = env.reset()
    #obs = resize(rgb2gray(env.reset()),(80,80))
    #obs = obs[np.newaxis, :, :]

    reward = 0
    done = False
    R = 0

    while not done:
        action = agent.act(obs)
        #action = agent.act_and_train(obs, reward)
        #action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        #obs = resize(rgb2gray(obs), (80, 80))
        #obs = obs[np.newaxis, :, :]


    agent.stop_episode()


last_time = datetime.datetime.now()

filename = "toreplace"
env.set_window(False)


print("Starting the training!")

for epoch in range(n_epochs):
    print("\nEpoch %d\n-------" % (epoch + 1))
    train_episodes_finished = 0
    train_scores = []
    reward = 0
    done = False

    print("Training...")
    obs = env.reset()
    for learning_step in trange(epoch_len, leave=False):
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        if done:
            score = env.get_total_score()
            train_scores.append(score)
            obs = env.reset()
            train_episodes_finished += 1

    print("%d training episodes played." % train_episodes_finished)

    train_scores = np.array(train_scores)
    agent.stop_episode_and_train(obs, score, done)
    print("Results: mean :",train_scores.mean()," plusminus " ,train_scores.std(), "min: ", train_scores.min(), "max: ", train_scores.max())

    print("\nTesting...")
    test_episode = []
    test_scores = []
    for test_episode in trange(test_epoch_len, leave=False):
        done = False
        obs = env.reset()
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
        r = env.get_total_score()
        test_scores.append(r)
        agent.stop_episode()

    test_scores = np.array(test_scores)
    print("Results: mean :",test_scores.mean()," plusminus " ,test_scores.std(), "min: ", test_scores.min(), "max: ", test_scores.max())

    #print("Saving the network weigths to:", model_savefile)
    #saver.save(session, model_savefile)

    print("Total elapsed time: ", ((datetime.datetime.now() - last_time) / 60.0), " minutes")


print("======================================")
print("Training finished. It's time to watch!")


"""


for i in range(1, n_epochs + 1):
    obs = env.reset()
    #obs = resize(rgb2gray(env.reset()),(80,80))
    #obs = obs[np.newaxis, :, :]
    t = 0
    reward = 0
    done = False
    R = 0

    while t<epoch_len:
        action = agent.act_and_train(obs, reward)
        obs, reward, done, _ = env.step(action)
        #obs = resize(rgb2gray(obs), (80, 80))
        #obs = obs[np.newaxis, :, :]
        t = t + 1
        if reward != 0:
            R += reward
        if done:
            obs = env.reset()
            done=False

    elapsed_time = datetime.datetime.now() - last_time
    print('epoch:', i, '/', n_epochs,
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
"""


env.set_window(True)

for i in range(1, 300 + 1):
    obs = env.reset()
    #obs = resize(rgb2gray(env.reset()),(80,80))
    #obs = obs[np.newaxis, :, :]

    reward = 0
    done = False
    R = 0

    while not done:
        action = agent.act(obs)
        #action = agent.act_and_train(obs, reward)
        #action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        #obs = resize(rgb2gray(obs), (80, 80))
        #obs = obs[np.newaxis, :, :]


    agent.stop_episode()
print("demo ended")


