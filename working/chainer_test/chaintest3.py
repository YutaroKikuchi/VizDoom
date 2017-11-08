# example from https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb modified for doom
from __future__ import print_function
from vizdoom import *

from random import choice
from time import sleep

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np

from skimage.color import rgb2gray
from skimage.transform import resize

scenario = "./config/basic.wad"
dmap = "map01"

# Create DoomGame instance. It will run the game and communicate with you.
game = DoomGame()

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
# game.load_config("../../scenarios/basic.cfg")

# Sets path to additional resources wad file which is basically your scenario wad.
# If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
game.set_doom_scenario_path(scenario)

# Sets map to start (scenario .wad files can contain many maps).
game.set_doom_map(dmap)

# Sets resolution. Default is 320X240
game.set_screen_resolution(ScreenResolution.RES_640X480)

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
game.set_screen_format(ScreenFormat.RGB24)

# Enables depth buffer.
game.set_depth_buffer_enabled(True)

# Enables labeling of in game objects labeling.
game.set_labels_buffer_enabled(True)

# Enables buffer with top down map of the current episode/level.
game.set_automap_buffer_enabled(True)

# Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
game.set_render_hud(False)
game.set_render_minimal_hud(False)  # If hud is enabled
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)  # Bullet holes and blood on the walls
game.set_render_particles(False)
game.set_render_effects_sprites(False)  # Smoke and blood
game.set_render_messages(False)  # In-game messages
game.set_render_corpses(False)
game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

# Adds buttons that will be allowed. 
game.add_available_button(Button.MOVE_LEFT)
game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.ATTACK)

# Adds game variables that will be included in state.
game.add_available_game_variable(GameVariable.AMMO2)

# Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(200) #2000

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
game.set_window_visible(False)


# Turns on the sound. (turned off by default)
game.set_sound_enabled(True)

# Sets the livin reward (for each move) to -1
game.set_living_reward(-1)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game.set_mode(Mode.PLAYER)

# Enables engine output to console.
#game.set_console_enabled(True)

print("--basic setup done--")

my_game = game
# Initialize the game. Further configuration won't take any effect from now on.
my_game.init()

# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.
actions = [[True, False, False], [False, True, False], [False, False, True]]
actions_int = [0, 1, 2]

# Run this many episodes
episodes = 10

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
sleep_time = 1.0 / DEFAULT_TICRATE # = 0.028

# env from tutorial
#env = gym.make('CartPole-v0')

#obs = env.reset()
#env.render()

action = actions # env.action_space.sample()

"""
class QFunction(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(QFunction, self).__init__(
            # the size of the inputs to each layer will be inferred
            l1=L.Linear(None, n_units),  # n_in -> n_units
            l2=L.Linear(None, n_units),  # n_units -> n_units
            l3=L.Linear(None, n_out),  # n_units -> n_out
        )
 
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y
        """
class QFunction(chainer.Chain):
    def __init__(self, n_history=1, n_action=3):
        super().__init__(
            l1=L.Convolution2D(n_history, 32, ksize=8, stride=4, nobias=False),
            l2=L.Convolution2D(32, 64, ksize=3, stride=2, nobias=False),
            l3=L.Convolution2D(64, 64, ksize=3, stride=1, nobias=False),
            l4=L.Linear(1792, 512),
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

#obs_size = env.observation_space.shape[0]
n_actions = len(actions_int) #env.action_space.n

obs_size = 640 * 480
# 307200 * 3
# 921600
obs_size = 921600
unit = 50                 # Number of hidden layer units, try incresing this value and see if how accuracy changes.

#q_func = QFunction(obs_size, 10)
q_func = QFunction(1, n_actions)

# Uncomment to use CUDA
# q_func.to_gpu(0)

"""
_q_func = chainerrl.q_functions.FCStateQFunctionWithDiscreteAction(
    obs_size, n_actions,
    n_hidden_layers=2, n_hidden_channels=50)
    
"""

# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)


# Set the discount factor that discounts future rewards.
gamma = 0.95

def random_action_doom() :
    result = choice(actions_int)
    return result

# TODO : replace env.action_space.sample (random action function)
# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=random_action_doom)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)

# Since observations from CartPole-v0 is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
"""
agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100, phi=phi)
"""


agent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    minibatch_size=4, replay_start_size=500,
     phi=phi)

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    my_game.new_episode() # obs reset
    state = game.get_state()
    img = state.screen_buffer
    #print(len(img))
    obs = img#.flatten()#.tolist()
    obs = resize(rgb2gray(obs), (80, 60))
    obs = obs[np.newaxis, :, :]
    #print("-------------------print obs------------------------")
    #print(type(obs))
    #print(len(obs))
    #print(len(obs[0]))
    #print(len(obs[0][0]))
    #print(obs)
    reward = 0
    R = 0  # return (sum of rewards)
    t = 0  # time step
    save_every = 50
    
    while not my_game.is_episode_finished():
        # Uncomment to watch the behaviour
        # env.render()
        
        # TODO : replace obs by image
        action = agent.act_and_train(obs, reward)
        
        state = game.get_state()
        img = state.screen_buffer
        """"obs
        <class 'numpy.ndarray'>
        [-0.02013112  0.02886185  0.0183539  -0.03637876]
        """
        
        obs = img#.flatten()#.tolist()
        obs = resize(rgb2gray(obs), (80, 60))
        obs = obs[np.newaxis, :, :]
               

        reward = game.make_action(actions[action])
        R += reward
        
        t += 1
        
        # Which consists of:
        n = state.number
        vars = state.game_variables

        # Prints state's game variables and reward.
        #print("State #" + str(n))
        #print("Game variables:", vars)
        if t%100==0:
            print("Reward:", reward)
            print("Action :", action)
            print("Episode :", i, "/", episodes)
        #print("=====================")

        #if sleep_time > 0:
            #sleep(sleep_time)
        
    agent.stop_episode_and_train(obs, reward, True)
     # Check how the episode went.
    print("Episode finished.", 'statistics:', agent.get_statistics())
    print("Total reward:", game.get_total_reward())
    print("************************")
    if i % save_every == 0:
        filename = '/home/vizdoom/agent' + str(i)
        #agent.save(filename)
print('Finished.')

# Save an agent to the 'agent' directory
#agent.save('/home/vizdoom/agent')

#agent.load('/home/vizdoom/agent')
print("begin test")

game.close()
game.set_window_visible(True)
my_game.init()

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    my_game.new_episode() # obs reset
    state = game.get_state()
    img = state.screen_buffer
    print(len(img))
    obs = img#.flatten()#.tolist()
    obs = resize(rgb2gray(obs), (80, 60))
    obs = obs[np.newaxis, :, :]
    print("-------------------print obs------------------------")
    print(type(obs))
    print(len(obs))
    print(len(obs[0]))
    print(len(obs[0][0]))
    print(obs)
    reward = 0


    R = 0  # return (sum of rewards)
    t = 0  # time step
    
    
    while not my_game.is_episode_finished():

        action = agent.act(obs)
        
        state = game.get_state()
        img = state.screen_buffer
        
        obs = img#.flatten()#.tolist()
        obs = resize(rgb2gray(obs), (80, 60))
        obs = obs[np.newaxis, :, :]
               

        reward = game.make_action(actions[action])

        if sleep_time > 0:
            sleep(sleep_time)
        
    agent.stop_episode()
     # Check how the episode went.
    print("Episode finished.", 'statistics:', agent.get_statistics())
    print("Total reward:", game.get_total_reward())
    print("************************")
print('TEST Finished.')





