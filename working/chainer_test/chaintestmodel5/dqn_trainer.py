# modified https://github.com/icoxfog417/chainer_pong/blob/master/model/dqn_trainer.py
import numpy as np
from chainer import Variable
from chainer import optimizers
from chainer import cuda
from chaintestmodel5.dqn_agent import Q
import chainer.functions as F
from chaintestmodel5.agent import Agent


class DQNTrainer(Agent):
    
    def __init__(self, 
                    agent, 
                    memory_size=10**4,
                    replay_size=32,
                    gamma=0.99,
                    #initial_exploration=10**2, #small value for test only
                    initial_exploration=10**4, #real value
                    target_update_freq=10**4,
                    learning_rate=0.00025,
                    epsilon_decay=1e-6,
                    minimum_epsilon=0.1):
        self.agent = agent
        self.target = Q(self.agent.q.n_history, self.agent.q.n_action, on_gpu=self.agent.q.on_gpu)

        self.memory_size = memory_size
        self.replay_size = replay_size
        self.gamma = gamma
        self.initial_exploration = initial_exploration
        self.target_update_freq = target_update_freq
        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self._step = 0

        # prepare memory for replay
        n_hist = self.agent.q.n_history
        sizex = self.agent.q.sizex
        sizey = self.agent.q.sizey
        self.memory = [
            np.zeros((memory_size, n_hist, sizex, sizey), dtype=np.float32),
            np.zeros(memory_size, dtype=np.uint8),
            np.zeros((memory_size, 1), dtype=np.float32),
            np.zeros((memory_size, n_hist, sizex, sizey), dtype=np.float32),
            np.zeros((memory_size, 1), dtype=np.bool)
        ]
        self.memory_text = [
            "state", "action", "reward", "next_state", "episode_end"
        ]

        # prepare optimizer
        self.optimizer = optimizers.RMSpropGraves(lr=learning_rate, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.agent.q)
        self._loss = 9
        self._qv = 0

    def calc_loss(self, states, actions, rewards, next_states, episode_ends):
        print("calculate loss, states :",len(states), "next stqtes:", len(next_states))
        #print(states)
        qv = self.agent.q(states)
        """
        qv = np.ndarray(shape = (0,3), dtype = "float32")
        ii=0
        for item in states:
            obsitem = np.ndarray(shape = (1,4,80,80), dtype = "float32")
            obsitem[0] = item
            a =(self.target(obsitem))[0]
            b =a.array
            c = np.ndarray(shape = (3), dtype = "float32")
            for e in range(len(b)):
                c[e]=b[e]
            qvbis = np.ndarray(shape = (1,3), dtype = "float32")
            qvbis[0]=c
            qv = np.concatenate((qv,qvbis))
            ii = ii + 1
        #print(qv)
        qv = Variable(data=qv)
        #print(qv)"""
        
        
        q_t = self.target(next_states)  # Q(s', *)
        
        """q_t = np.ndarray(shape = (0,3), dtype = "float32")
        ii=0
        for item in next_states:
            obsitem = np.ndarray(shape = (1,4,80,80), dtype = "float32")
            obsitem[0] = item
            a =(self.target(obsitem))[0]
            b =a.array
            c = np.ndarray(shape = (3), dtype = "float32")
            for e in range(len(b)):
                c[e]=b[e]
            qvbis = np.ndarray(shape = (1,3), dtype = "float32")
            qvbis[0]=c
            q_t = np.concatenate((q_t,qvbis))
            ii = ii + 1
        #print(q_t)
        q_t = Variable(data=q_t)
        #print(q_t)"""
        
        
        #print("calculate loss, qv :",type(qv), "qt:", type(q_t))
        #print("dty, qv :",qv.dtype, "qt:", q_t.dtype)
        #print("dty, qv :",qv.data.dtype, "qt:", q_t.data.dtype)
        #print("dty, qv :",qv.data[0].dtype, "qt:", q_t.data[0].dtype)
        #print("values, qv :",type(qv.data), "qt:", type(q_t.data))
        #print("alltypes",type(qv),type(qv.data),type(qv.data[0]),type(qv.data[0][0]))
        #print("alltypes",type(q_t),type(q_t.data),type(q_t.data[0]),type(q_t.data[0][0]))
        #print("alltypes",len(qv),len(qv.data),len(qv.data[0]))
        #print("alltypes",len(q_t),len(q_t.data),len(q_t.data[0]))
        #print(qv)
        #print(q_t)
        max_q_prime = np.array(list(map(np.max, q_t.data)), dtype=np.float32)  # max_a Q(s', a)
        
        target = cuda.to_cpu(qv.data.copy())
        for i in range(self.replay_size):
            if episode_ends[i][0] is True:
                _r = np.sign(rewards[i])
            else:
                _r = np.sign(rewards[i]) + self.gamma * max_q_prime[i]
            
            target[i, actions[i]] = _r
        
        td = Variable(self.target.arr_to_gpu(target)) - qv
        #print(td.data)
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zeros = Variable(self.target.arr_to_gpu(np.zeros((self.replay_size, self.target.n_action), dtype=np.float32)))
        loss = F.mean_squared_error(td_clip, zeros)
        self._loss = loss.data
        self._qv = np.max(qv.data)
        return loss

    def start(self, observation):
        return self.agent.start(observation)
    
    def act(self, observation, reward, framefirstorlast=False):
        if self.initial_exploration <= self._step:
            self.agent.epsilon -= 1.0/10**6
            if self.agent.epsilon < self.minimum_epsilon:
                self.agent.epsilon = self.minimum_epsilon
        
        return self.train(observation, reward, episode_end=False, framefirstorlast=framefirstorlast)

    def end(self, observation, reward):
        self.train(observation, reward, episode_end=True)

    def train(self, observation, reward, episode_end, framefirstorlast=False):
        action = 0
        last_state = self.agent.get_state()
        last_action = self.agent.last_action
        if not episode_end:
            action = self.agent.act(observation, reward, framefirstorlast=framefirstorlast)
            result_state = self.agent.get_state()
            self.memorize(
                last_state, 
                last_action, 
                reward,
                result_state,
                False)
        else:
            self.memorize(
                last_state, 
                last_action, 
                reward,
                last_state,
                True)
        
        if self.initial_exploration <= self._step:
            self.experience_replay()

            if self._step % self.target_update_freq == 0:
                self.target.copyparams(self.agent.q)

        self._step += 1
        return action

    def memorize(self, state, action, reward, next_state, episode_end):
        _index = self._step % self.memory_size
        self.memory[0][_index] = state
        self.memory[1][_index] = action
        self.memory[2][_index] = reward
        if not episode_end:
            self.memory[3][_index] = next_state
        self.memory[4][_index] = episode_end

    def experience_replay(self):
        indices = []
        if self._step < self.memory_size:
            indices = np.random.randint(0, self._step, (self.replay_size))
        else:
            indices = np.random.randint(0, self.memory_size, (self.replay_size))
        
        states = []
        actions = []
        rewards = []
        next_states = []
        episode_ends = []
        for i in indices:
            states.append(self.memory[0][i])
            actions.append(self.memory[1][i])
            rewards.append(self.memory[2][i])
            next_states.append(self.memory[3][i])
            episode_ends.append(self.memory[4][i])
        
        to_np = lambda arr: np.array(arr)
        self.optimizer.target.cleargrads()
        loss = self.calc_loss(to_np(states), to_np(actions), to_np(rewards), to_np(next_states), to_np(episode_ends))
        loss.backward()
        self.optimizer.update()
    
    def report(self, episode):
        s = "{0}: loss={1}, q value={2}, epsilon={3}".format(self._step, self._loss, self._qv, self.agent.epsilon)
        self.agent.save(episode)
        return s