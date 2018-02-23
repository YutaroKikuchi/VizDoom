import numpy as np
import math


class ReplayBuffer():
    # where we memorize past scenarios

    def __init__(self,capacity=10**4,sizex=80,sizey=80, nb_action=1):
        # -- initialize the buffer
        # capacity : defines how many scenarios we can store
        # sizex : width of the images to store
        # sizey : height of the images to store
        
        # memorize settings
        self.before_action_obs=np.zeros((capacity, sizex, sizey), dtype=np.float32)
        self.after_action_obs=np.zeros((capacity, sizex, sizey), dtype=np.float32)
        if nb_action>1:
            self.action=np.zeros((capacity, nb_action))
        else :
            self.action=np.zeros(capacity, dtype=np.uint8)
        self.reward=np.zeros((capacity, 1), dtype=np.float32)
        self.is_episode_end=np.zeros((capacity, 1), dtype=np.bool)
        self.cursor=0
        self.capacity=capacity
    
    
    def stock_replay_information(self, obs1, action, obs2, reward, end_episode):
        # -- add a new scenario to the buffer
        # obs1 : observation before the action
        # action : action that was chosen
        # obs2 : observation after the action
        # reward : reward we got after accomplishing the action
        # end_episode : if it was the end of the episode (usually obs1=obs2)
        
        # memorize the data at the right position
        self.before_action_obs[self.cursor]=obs1
        self.after_action_obs[self.cursor]=obs2
        self.action[self.cursor]=action
        self.reward[self.cursor]=reward
        self.is_episode_end[self.cursor]=end_episode
        self.cursor=self.cursor+1
        
        # if reached the max capacity, overwrite the oldest data
        if self.capacity==self.cursor:
            self.cursor=0

