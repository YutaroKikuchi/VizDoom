
import numpy as np
import math


class ReplayBuffer():

    def __init__(self,capacity=10**4,n_hist=64,sizex=80,sizey=80):
        # show_frames set to True to see the first and last frame of each epoch with the 3 convolutions
        self.before_action_obs=np.zeros((capacity, n_hist, sizex, sizey), dtype=np.float32)
        self.after_action_obs=np.zeros((capacity, n_hist, sizex, sizey), dtype=np.float32)
        self.action=np.zeros(capacity, dtype=np.uint8)
        self.reward=np.zeros((capacity, 1), dtype=np.float32)
        self.is_episode_end=np.zeros((capacity, 1), dtype=np.bool)
        self.cursor=0
        self.capacity=capacity
    
    def stock_replay_information(self, obs1, action, obs2, reward, end_episode):
        self.before_action_obs[self.cursor]=obs1
        self.after_action_obs[self.cursor]=obs2
        self.action[self.cursor]=action
        self.reward[self.cursor]=reward
        self.is_episode_end[self.cursor]=end_episode
        self.cursor=self.cursor+1
        if self.capacity==self.cursor:
            self.cursor=0
    
#    self.cursor  self.before_action_obs[self.cursor]  self.after_action_obs[self.cursor] self.action[self.cursor] self.reward[self.cursor] self.is_episode_end[self.cursor]=end_episode