# example from https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb modified for doom
# and combined with https://github.com/icoxfog417/chainer_pong/blob/master/model/environment.py
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

import matplotlib.pyplot as plt

from tqdm import trange

from skimage.color import rgb2gray
from skimage.transform import resize

n_epochs = 20 #20
epoch_len = 1000
test_epoch_len =8
save_every = 5


class Environment(env.Env):

    def __init__(self, scenario="./config/basic.wad", dmap = "map01", episode_len=200, window=True, show_frames =False):
        # show_frames set to True to see the first and last frame of each epoch with the 3 convolutions
        self.show_frames=show_frames
        self.game = DoomGame()
        self.game.set_doom_scenario_path(scenario)
        self.game.set_doom_map(dmap)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_screen_format(ScreenFormat.GRAY8) #GRAY8 RGB24
        self.game.set_depth_buffer_enabled(False)
        self.game.set_labels_buffer_enabled(False)
        self.game.set_automap_buffer_enabled(False)
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
        obs = resize(self.game.get_state().screen_buffer, (80, 80))
        obs = obs[np.newaxis, :, :]
        self.current_screen = obs
        
        # adapt to other example
        #self.env = self.game
        self.actions = self.legal_actions
        
    # TODO: modify this part
    def play(self, agent, epochs=5, render=True, report_interval=-1, action_interval=1, record_path=""):
        scores = []
        self.set_window(render)

        epoch_len = 1000
        
        all_rewards=[]
        all_means=[]
        
        for i in range(epochs):
            print("Epoch nb " + str(i))
            observation = self.reset()
            done = False
            reward = 0.0
            step_count = 0
            score = 0.0
            continue_game = True
            last_action = 0
            scores = []
            while step_count<epoch_len:

                if step_count == 0:
                    action = agent.start(observation)
                else:
                    if step_count % action_interval == 0 or reward != 0:
                        action = agent.act(observation, reward, framefirstorlast=(self.show_frames and (step_count==epoch_len-1))) # step_count==1 or
                    else:
                        action = last_action

                observation, reward, done, info = self.step(action)
                
                last_action = action

                if done:
                    agent.end(observation, reward)
                
                yield i, step_count, reward

                continue_game = not done
                score += reward
                step_count += 1
                
                if((not continue_game) or (step_count+1==epoch_len)):
                    scores.append(score)
                    score = 0
                    observation = self.reset()
                    continue_game = True
            
            print("average score is {0}.".format(sum(scores) / len(scores)))
            all_means.append(sum(scores) / len(scores))
            report = agent.report(i)
            all_rewards = all_rewards + scores
                

            
        all_rewards = np.array(all_rewards)
        print(type(all_rewards))
        plt.plot(all_rewards)
        plt.savefig("graph_epoch_"+ "all" +".png")
        plt.show()


        all_means = np.array(all_means)
        print(type(all_means))
        plt.plot(all_means)
        plt.savefig("graph_epoch_"+ "allmeans" +".png")
        plt.show()
            


    @property
    def state(self):
        #print(self.game.get_screen_format())
        
        if self.game.is_episode_finished():
            return self.current_screen
        rr = self.game.get_state()
        
        #obs=rr.screen_buffer
        obs = resize(rr.screen_buffer, (80, 80))
        #print(obs)
        render = obs
        obs = obs[np.newaxis, :, :]
        self.current_screen = obs
        
        #plt.imshow(render)
        #titless = plt.title('just check')
        #plt.show()
        
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
        self._reward = self.game.make_action(self.legal_actions[action], 10)
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
