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

# Run this many episodes
episodes = 15
episode_len = 200

# ---------------------------------------------------------------------------------DOOM_config_beginning
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
game.set_episode_timeout(episode_len) #2000

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
game.set_window_visible(True)


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

# Sets time that will pause the engine after each action (in seconds)
# Without this everything would go too fast for you to keep track of what's happening.
sleep_time = 1.0 / DEFAULT_TICRATE # = 0.028

# ---------------------------------------------------------------------------------DOOM_config_end

action = actions # env.action_space.sample()


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
   

n_actions = len(actions_int) #env.action_space.n

q_func = QFunction(1, n_actions)

# Uncomment to use CUDA
# q_func.to_gpu(0)


# Use Adam to optimize q_func. eps=1e-2 is for stability.
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)

# Set the discount factor that discounts future rewards.
gamma = 0.95

def random_action_doom() :
    result = choice(actions_int)
    return result

# Use epsilon-greedy for exploration
explorer = chainerrl.explorers.ConstantEpsilonGreedy(
    epsilon=0.3, random_action_func=random_action_doom)

# DQN uses Experience Replay.
# Specify a replay buffer and its capacity.
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 4)

# Since observations is numpy.float64 while
# Chainer only accepts numpy.float32 by default, specify
# a converter as a feature extractor function phi.
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.

sagent = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    minibatch_size=4, replay_start_size=500,
     phi=phi)

for i in range(episodes):
    print("Episode #" + str(i + 1))

    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    my_game.new_episode() # obs reset
    state = game.get_state()
    img = state.screen_buffer

    obs = img#.flatten()#.tolist()
    obs = resize(rgb2gray(obs), (80, 80))
    obs = obs[np.newaxis, :, :]

    reward = 0
    R = 0  # return (sum of rewards)
    t = 0  # time step
    save_every = 50
    
    while not my_game.is_episode_finished():
        action = sagent.act_and_train(obs, reward)
        
        state = game.get_state()
        img = state.screen_buffer
        
        obs = img
        obs = resize(rgb2gray(obs), (80, 80))
        obs = obs[np.newaxis, :, :]
        
        reward = game.make_action(actions[action])
        R += reward
        t += 1
        
        if t%100==0:
            print("Reward:", reward)
            print("Action :", action)
            print("Episode :", i, "/", episodes)

        #if sleep_time > 0:
            #sleep(sleep_time)
        
    sagent.save('agent')
    sagent.stop_episode_and_train(obs, reward, True)
    # Check how the episode went.
    print("Episode finished.", 'statistics:', sagent.get_statistics())
    print("Total reward:", game.get_total_reward())
    print("************************")
    if i % save_every == 0:
        filename = '/home/vizdoom/agent' + str(i)
        #agent.save(filename)
print('Finished.')

# Save an agent to the 'agent' directory

sagent.load('agent')
print("begin test")

game.close()
game.set_window_visible(True)
my_game.init()

for i in range(episodes):
    print("Episode #" + str(i + 1))

    my_game.new_episode() # obs reset
    state = game.get_state()
    img = state.screen_buffer
    
    obs = img
    obs = resize(rgb2gray(obs), (80, 80))
    obs = obs[np.newaxis, :, :]
    reward = 0

    R = 0  # return (sum of rewards)
    t = 0  # time step
    
    
    while not my_game.is_episode_finished():

        action = sagent.act(obs)
        #action = agent.act_and_train(obs, reward)
        
        state = game.get_state()
        img = state.screen_buffer
        obs = img
        obs = resize(rgb2gray(obs), (80, 80))
        obs = obs[np.newaxis, :, :]

        reward=game.make_action(actions[action])

        if sleep_time > 0:
            sleep(sleep_time)
        
    sagent.stop_episode()
    
    print("Episode finished.", 'statistics:', sagent.get_statistics())
    print("Total reward:", game.get_total_reward())
    print("************************")
print('TEST Finished.')





