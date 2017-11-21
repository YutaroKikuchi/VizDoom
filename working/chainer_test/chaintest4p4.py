# example from https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb modified for doom
from __future__ import print_function
from vizdoom import *

from random import choice
from time import sleep

import datetime
import chainer
import chainer.functions as F
import chainer.links as L
#from chainer import cuda
import chainerrl
from chainerrl import env
from chainerrl import spaces
import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.transform import resize

n_episodes = 10
save_every = 5
steps_per_epoch = 1000

#cuda.check_cuda_available()
#xp = chainer.cuda.cupy

class DOOM_ENV(env.Env):
    """Arcade Learning Environment."""

    def __init__(self, scenario="./config/basic.wad", dmap = "map01", episode_len=200, window=False):
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
        self._reward = self.game.make_action(self.legal_actions[action], tics = 10) # do an action every 10 frames
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

    def set_visible(self, visible):
        self.game.set_window_visible(visible)

    def close(self):
        pass

env = DOOM_ENV()
obs = env.reset()

"""
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

"""

obs_size = 80*80
n_actions = 3
    

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=50):
        #super(QFunction, self).__init__(##python2.x用
        super().__init__(#python3.x用
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels,n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))
        
    def __call__(self, x, test=False): 
        h = F.tanh(self.l0(x)) #活性化関数は自分で書くの？
        h = F.tanh(self.l1(h))
        return chainerrl.action_value.DiscreteActionValue(self.l2(h))



    
    
n_action = env.number_of_actions
print("n action size obs")
print(n_action)
n_history=1
q_func = QFunction(obs_size, n_action)

optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

gamma = 0.95

explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=env.random_action_doom)#epsilon 1.0 better ?!

replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

phi = lambda x: x.astype(np.float32, copy=False)

agent =chainerrl.agents.DoubleDQN(
   q_func, optimizer, replay_buffer, gamma, explorer,
   replay_start_size=500, update_interval=1,
   target_update_interval=100, phi=phi)
"""
chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    minibatch_size=4, replay_start_size=300,
     phi=phi)
"""

last_time = datetime.datetime.now()

env.set_visible(False)

gpu_device = 0
#cuda.get_device(gpu_device).use()
#q_func.to_gpu(gpu_device)

all_rewards = []

filename = "toreplace"
for i in range(1, n_episodes + 1):

    
    print("Epoch%d\n________"%(i))
    #obs = resize(rgb2gray(env.reset()),(80,80))
    #obs = obs[xp.newaxis, :, :]
    #print(obs)
    #print(type(obs[0][0][0]))
    #obs = image2gpu(obs,gpu_device)
    #obs = chainer.cuda.to_gpu(obs,device=gpu_device)
    #obs = obs.astype(cp.float32)
    #print(type(obs[0][0][0]))
    #print(type(obs))
    #print(obs.shape)
    #print(obs)
    #obs = cp.ndarray(obs.tolist())

    obs = env.reset()
    reward = 0
    done = False
    R = 0

    trained_episode = 1
    total_reward = []

    last_time = datetime.datetime.now()
    for step in tqdm(range(steps_per_epoch)):
        #if step > 490:
        #  print(type(obs))
        
        action = agent.act_and_train(obs,reward)
        obs, reward, done, _ = env.step(action)
        #obs = resize(rgb2gray(obs), (80, 80))
        #obs = obs[xp.newaxis, :, :]

        if reward != 0:
            R += reward

        if done == True:
            agent.stop_episode_and_train(obs, reward, done)
            #obs = resize(rgb2gray(env.reset()),(80,80))
            #obs = obs[xp.newaxis, :, :]
            trained_episode += 1
            total_reward.append(R)
            obs = env.reset()
            reward = 0
            done = False
            R=0

    elapsed_time = datetime.datetime.now() - last_time
    print("%d episodes is trained")
    print('minutes:', elapsed_time.seconds/60)
    total_reward = np.array(total_reward)
    print("reward_mean:%d, reward_max:%d, reward_min:%d" % (np.mean(total_reward), np.max(total_reward), np.min(total_reward)))

    agent.stop_episode_and_train(obs, reward, done)
    if i % save_every == 0:
        filename = 'agent_Breakout' + str(i)
        agent.save(filename)
    
    gra = plt.plot(total_reward)
    gra.set_ylabel('rewards per episode')
    gra.savefig("graph_epoch_"+ i +".png")
    
    all_rewards +=total_reward
    
print('Finished. Now testing')

print("demo starts")
#agent.load(filename)

gra = plt.plot(all_rewards)
gra.set_ylabel('rewards per episode')
gra.savefig("graph_epoch_"+ "all" +".png")

env.set_window(True)
for i in range(20):
    #obs = resize(rgb2gray(env.reset()),(80,80))
    #obs = obs[xp.newaxis, :, :]
    obs = env.reset()
    reward = 0
    done = False
    R = 0

    while not done:
        action = agent.act(obs)
        #action = agent.act_and_train(obs, reward)
        #action = agent.act(obs)
        obs, reward, done, _ = env.step(action)
        #obs = resize(rgb2gray(obs), (80, 80))
        #obs = obs[xp.newaxis, :, :]
        #obs = chainer.cuda.to_gpu(obs, device=gpu_device)
        print(i)


    agent.stop_episode()
print("demo ended")
