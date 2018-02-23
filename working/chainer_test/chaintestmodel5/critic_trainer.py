# modified https://github.com/icoxfog417/chainer_pong/blob/master/model/dqn_trainer.py
import numpy as np
from chainer import Variable
from chainer import optimizers
from chainer import cuda
from chaintestmodel5.dqn_agent import Q
from chaintestmodel5.critic_agent import QCritic
from chaintestmodel5.replay_buffer import ReplayBuffer
import chainer.functions as F
from chaintestmodel5.agent import Agent
from threading import Thread


class CriticTrainer(Agent):
    # the one organizing the training
    # replay_size=64
    
    def __init__(self, agent, memory_size=10**4, replay_size=64, gamma=0.99, initial_exploration=65, target_update_freq=10**4, learning_rate=0.00025,
                    minimum_epsilon=0.1):
        # -- Initialize the trainer with the given settings
        # agent : the agent to train, it is the one doing the thinking and deciding which action to do
        # memory_size : memory size for past observations with their action and consequence
        # replay_size : how many past scenarios should be re-experienced during the replay phase
        # gamma : constant to calculate the loss
        # initial_exploration : number of frames before doing replays, should be higher than replay_size in order to have enough scenarios for replay
        # target_update_freq : every multiple of steps of this number, reset the two Q functions to be equal to eachother
        # learning_rate : optimizer parameter
        # minimum_epsilon : minimum value of epsilon (epsilon decreases until it reaches this value)
        
        # memorize settings
        self.agent = agent
        if agent.on_gpu :
            self.target = QCritic(self.agent.q.n_channels, self.agent.q.n_action, on_gpu=self.agent.q.on_gpu).to_gpu()
        else:
            self.target = QCritic(self.agent.q.n_channels, self.agent.q.n_action, on_gpu=self.agent.q.on_gpu)
        self.memory_size = memory_size
        self.replay_size = replay_size
        self.gamma = gamma
        self.initial_exploration = initial_exploration
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        self.minimum_epsilon = minimum_epsilon
        self._step = 0
        self.sub_step=0
        self.loss_values =[]
        
        # prepare memory for replay
        n_hist = self.agent.q.n_channels
        sizex = self.agent.q.sizex
        sizey = self.agent.q.sizey
        self.memory=ReplayBuffer(memory_size, sizex, sizey, 2)

        # prepare optimizer
        self.optimizer_critic = optimizers.RMSpropGraves(lr=learning_rate, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer_critic.setup(self.agent.q_critic)
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.agent.q)
        self._loss = 9
        self._qv = 0

        
    def calc_loss(self, indices):
        # -- calculate the loss : E[(reward + gamma*maxQ' - Q)**2]
        # indices : the index of the past scenarios from the memory buffer
        
        # prepare (Q)
        to_np = lambda arr: np.array(arr)
        states=to_np([[self.memory.before_action_obs[i]] for i in indices])
        old_action=to_np([np.zeros(self.agent.q_critic.n_action, dtype=np.float32) for i in indices])
        for act in range(len(indices)):
            old_action[act]=self.memory.action[indices[act]]
        #print("old action ", old_action)
        qv = self.agent.q_critic(states,old_action) # Q
        # prepare (maxQ')
        next_states=to_np([[self.memory.after_action_obs[i]] for i in indices])
        q_t = self.target(next_states,old_action)
        max_q_prime = np.array(list(map(np.max, q_t.data)), dtype=np.float32)  # maxQ'
        
        # prepare (reward + gamma*maxQ')
        #print("qvdata ",qv.data)
        target = cuda.to_cpu(qv.data.copy())
        for i in range(self.replay_size):
            if self.memory.is_episode_end[i][0] is True:
                _r = np.sign(self.memory.reward[indices[i]])
            else:
                _r = np.sign(self.memory.reward[indices[i]]) + self.gamma * max_q_prime[i]
                #print("target",target)
            target[i][0] = _r
        td = Variable(self.target.arr_to_gpu(target)).data

        loss = F.mean_squared_error(td, qv) # E[(reward + gamma*maxQ' - Q)**2]
        # update attributes
        self._loss = loss.data
        self._qv = np.max(qv.data)
        
        return loss
    
    
    def start(self, observation):
        Thread(target=self.experience_replay).start()
        return self.agent.start(observation, repeat= 1)
    
    
    def act(self, observation, reward, framefirstorlast=False):
        # -- get best next action and sometimes explore random actions
        # observation : image from which we will decide what the next action should be
        # reward : here useful for the training phase
        # framefirstorlast : only used to decide when to show the convolutions in case we chose to 
        #                                (in that case we show them during the end of an episode)
        
        # manage epsilon (decrease but not under minimum value)
        if self.initial_exploration <= self._step:
            self.agent.epsilon -= 1.0/10**6
            if self.agent.epsilon < self.minimum_epsilon:
                self.agent.epsilon = self.minimum_epsilon
        
        return self.train(observation, reward, episode_end=False, framefirstorlast=framefirstorlast)
    
    
    def end(self, observation, reward):
        self.train(observation, reward, episode_end=True)
    
    
    def train(self, observation, reward, episode_end, framefirstorlast=False):
        # -- do the training steps
        # observation : image from which we will decide what the next action should be
        # reward : last reward
        # episode_end : if it is the end of the episode
        # framefirstorlast : only used to decide when to show the convolutions in case we chose to 
        #                                (in that case we show them during the end of an episode)
        
        # set variables
        action = [0,False]
        last_state = self.agent.get_state()
        last_action = self.agent.last_action
        
        
        # decide next best action and memorize scenario in replay buffer
        if not episode_end:
            action = self.agent.act(observation, reward, framefirstorlast=framefirstorlast, repeat=1)
            #----actor-------------------
            s = self.agent.get_state()
            formated_observation=np.array([s])
            #print("action : ",action)
            #print("obs len : ",len(observation[0]))
            #print("reward : ",reward)
            #print("first type ",type(action[0]))
            action_nobool = action
            for i in range(len(action_nobool)):
                if isinstance(action_nobool[i], (bool,)):
                    action_nobool[i]  = 1 if action_nobool[i] else 0
            #array_action=np.array([np.zeros(self.agent.q_critic.n_action, dtype=np.float32)])
            #array_action[0][action]=1
            loss = self.agent.q_critic(formated_observation, np.array([action_nobool], dtype=np.float32))

            # update the optimizer (of the actor)
            loss.backward()
            self.optimizer.update()
            #----------------------------
            result_state = self.agent.get_state()
            self.memory.stock_replay_information(last_state, last_action, result_state[0], reward, False)
        else:
            self.memory.stock_replay_information(last_state, last_action, last_state, reward, True)
        
        # experience replays from replay buffer
        #if self.initial_exploration <= self._step:
            #self.experience_replay()
            # reset the Q functions to be equal
            #if self._step % self.target_update_freq == 0:
             #   self.target.copyparams(self.agent.q_critic)
        #self._step += 1
        
        return action
    
    
    def experience_replay(self):
        if self.initial_exploration <= self._step:
            # -- replay old scenarios
            # get random past scenarios from replay buffer
            indices = []
            if self._step < self.memory_size-1:
                indices = np.random.randint(0, self.memory.cursor, (self.replay_size))
            else:
                indices = np.random.randint(0, self.memory_size, (self.replay_size))

            #----critic------------------------
            # clear optimizer's gradiant
            self.optimizer_critic.target.cleargrads()
            # calculate loss
            to_np = lambda arr: np.array(arr)
            loss = self.calc_loss(to_np(indices))

            # memorize data for loss graph
            if self._step%10==0 :
                self.loss_values.append(loss.data)

            # update the optimizer (of the critic)
            loss.backward()
            self.optimizer_critic.update()
            
            if self._step % self.target_update_freq == 0:
                self.target.copyparams(self.agent.q_critic)
        self._step += 1
            
        
    
    
    def report(self, episode):
        s = "{0}: loss={1}, q value={2}, epsilon={3}".format(self._step, self._loss, self._qv, self.agent.epsilon)
        self.agent.save(episode)
        
        return s
