# example from https://github.com/chainer/chainerrl/blob/master/examples/quickstart/quickstart.ipynb modified for doom
# and combined with https://github.com/icoxfog417/chainer_pong/blob/master/model/environment.py
from __future__ import print_function
from vizdoom import *
from random import choice
from time import sleep
from time import gmtime, strftime
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


class Environment(env.Env):
    # where the Doom game takes place

    def __init__(self, scenario="./config/basic.wad", dmap = "map01", episode_len=200, render=True, show_frames =False, show_end_graph=True,
                 epoch_len = 1000):
        # -- Initialize the environment with the configurations you want
        # scenario : what Doom scenario to load (defines map, enemies` position, winning conditions...)
        # dmap : which map from the scenario we should use
        # episode_len : after how many frames the episode should stop if the winning condition hasn't been met
        # render : open the doom window during the play session
        # show_frames : True to see the first and last frame of each epoch with the 3 convolutions
        # show_end_graph : at the end show loss (if training) and score (all and epoch average)
        # epoch_len : how many frames should an epoch last (note : defines with episode_len the minimum number of episodes per epoch)
        
        # remember the configurations
        self.episode_len = episode_len
        self.epoch_len = epoch_len
        self.render=render
        self.show_end_graph = show_end_graph
        self.show_frames=show_frames
        # prepare Doom
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
        # sets the available actions
        self.game.add_available_button(Button.TURN_LEFT)
        self.game.add_available_button(Button.TURN_RIGHT)
        self.game.add_available_button(Button.ATTACK)
        self.legal_actions = [[True, False, False], [False, True, False], [False, False, True]]
        self.action_space = spaces.Discrete(len(self.legal_actions))
        self.actions = self.legal_actions
        # sets other game configs
        self.game.add_available_game_variable(GameVariable.AMMO2)
        self.game.set_episode_timeout(episode_len)
        self.game.set_episode_start_time(10)
        self.game.set_window_visible(render)
        self.game.set_sound_enabled(True)
        self.game.set_living_reward(-1)
        self.game.set_mode(Mode.PLAYER)
        # initialize the game and the important variables
        self.game.init()
        self.game.new_episode()
        self._reward=0
        self.frame = 0
        # first screenshot
        obs = resize(self.game.get_state().screen_buffer, (80, 80))
        obs = obs[np.newaxis, :, :]
        self.current_screen = obs
        
    
    def play(self, agent, epochs=5, render=True, action_interval=1):
        # -- main loop for training and testing
        # agent : the trainer (for training) or the agent (for testing)
        # epochs : number of epochs (the lengths are defined in the environment's initialization)
        # render : open the doom window during the play session
        # action_interval : the number of frames we repeat the same last action before changing
        
        # initialize array to store data for the graphs
        scores = []
        all_rewards=[]
        all_means=[]
        # configure window
        self.set_window(self.render)
        # epoch length to local variable, easier to read
        epoch_len = self.epoch_len
        
        # main loop (contains many epochs)
        for i in range(epochs):
            print("Epoch nb " + str(i))
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            # initialize game and variables for the new epoch
            observation = self.reset()
            episode_done = False
            reward = 0.0
            step_count = 0
            score = 0.0
            last_action = 0
            scores = []
            
            # epoch loop (contains many episodes)
            while step_count<epoch_len:
                if step_count == 0: # first step
                    action = agent.start(observation)
                else:
                    if step_count % action_interval == 0 or reward != 0:
                        # (possibly) change from previous action
                        action = agent.act(observation, reward, framefirstorlast=(self.show_frames and (step_count==epoch_len-1))) # step_count==1 or
                    else:
                        action = last_action # repeat previous action
                # act in the game and get data
                observation, reward, episode_done, info = self.step(action)
                last_action = action

                if episode_done: # if episode has ended because of a winning/losing condition met after doing the action
                    #print("another episode done ", step_count)
                    agent.end(observation, reward) # end the episode
                
                # update variables
                yield i, step_count, reward
                score += reward
                step_count += 1
                
                if(episode_done or (step_count+1==epoch_len)): # end of the episode
                    scores.append(score)
                    score = 0
                    observation = self.reset()
            
            print("average score is {0}.".format(sum(scores) / len(scores)))
            
            # update data for graphs
            all_means.append(sum(scores) / len(scores))
            report = agent.report(i)
            all_rewards = all_rewards + scores
                

        if self.show_end_graph: # show the graphs
            losses = np.array(agent.loss_values)
            print("losses ", type(losses))
            print(losses)
            print(np.shape(losses))
            all_rewards = np.array(all_rewards)
            print("all rewards ",type(all_rewards))
            print(np.shape(all_rewards))
            
            # show losses
            plt.plot(losses)
            plt.savefig("graph_epoch_"+ "losses" +".png")
            plt.show()
            # show total rewards of each episode
            plt.plot(all_rewards)
            plt.savefig("graph_epoch_"+ "all" +".png")
            plt.show()
            # show average score per epoch
            all_means = np.array(all_means)
            print(type(all_means))
            plt.plot(all_means)
            plt.savefig("graph_epoch_"+ "allmeans" +".png")
            plt.show()

    @property
    def state(self):
        # returns the current screen
        if self.game.is_episode_finished():
            return self.current_screen
        rr = self.game.get_state()
        obs = resize(rr.screen_buffer, (80, 80))
        render = obs
        obs = obs[np.newaxis, :, :]
        self.current_screen = obs
        return obs

    def random_action_doom(self,nothing=0) :
        # returns a random action
        result = choice(range(0,len(self.legal_actions)))
        return result

    @property
    def is_terminal(self):
        # returns True if last frame of episode or if winning/losing condition has been met
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
        self._reward = self.game.make_action(self.legal_actions[int(action)], 10)
        return self._reward

    def initialize(self):
        # initializes attributes for episodes
        self.game.new_episode()
        self._reward = 0
        self.frame = 0

    def reset(self):
        self.initialize()
        return self.state
    
    def set_window(self, on):
        # open or close the window
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
