# modified https://github.com/icoxfog417/chainer_pong/blob/master/model/dqn_agent.py
import os
import math
import numpy as np
from chainer import Chain
from chainer import Variable
from chainer import cuda
from chainer import serializers
import chainer.functions as F
import chainer.links as L
from chaintestmodel5.agent import Agent
import chainer.initializers as I 
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
from chaintestmodel5.dqn_agent import Q


class QCritic(Chain):
    # determines the next action to do from the given state
    
    
    # size of the given image (state)
    sizex = 80
    sizey = 80
    
    
    def __init__(self, n_channels, n_action, on_gpu=False):
        # -- initialize the model
        # n_channels : number of channels (1 for grayscale, 3 for rgb)
        # n_action : number of possible actions
        # on_gpu : can use gpu specific functions
        
        # memorize settings
        self.n_channels = n_channels
        self.n_action = n_action
        self.on_gpu = on_gpu
        if on_gpu:
            self.to_gpu()
        
        # define 2 convolutions, a linear function, a lstm (not used for now) and another linear function as the out
        # good parameters are : (n_channels, 8, ksize=6, stride=3, pad=1, nobias=False), (8, 8, ksize=3, stride=2, pad=1, nobias=False), 1352, 128
        super(QCritic, self).__init__(
            state_h1=L.Convolution2D(n_channels, 8, ksize=6, stride=3, pad=1, nobias=False,
                                  ),
            state_h2=L.Linear(5408, 48, initialW=I.HeNormal(np.sqrt(2)/ np.sqrt(2))),
            
            action_1=L.Linear(n_action, 48, initialW=I.HeNormal(np.sqrt(2)/ np.sqrt(2))), # if we do action 2 we give [0,0,1,0...]
            
            merged=L.Linear(6400, 24, initialW=I.HeNormal(np.sqrt(2)/ np.sqrt(2))),
            out=L.Linear(24, 1, initialW=np.zeros((1, 24), dtype=np.float32))) # gives the Q value used for calc_loss


    def __call__(self, state: np.ndarray, action: np.ndarray, show=False):
        # -- execute the q function on the current state
        # state : the current screen resized
        # show : show the convolutions
        
        # 1] prepare state
        _state = self.arr_to_gpu(state)
        h0 = Variable(_state)

        h1 = F.relu(self.state_h1(h0)) # 3 instead of 4
        if show:
            self.show_convolutions(h1)
        
        h2 = F.relu(self.state_h2(h1))
        if show:
            self.show_convolutions(h2)
        
        # 2] prepare action
        _action = self.arr_to_gpu(action)
        a0 = Variable(_action)
        
        a1 = F.relu(self.action_1(a0))
        
        # 3] merge
        _merge = self.arr_to_gpu(h2.data + a1.data)
        m0 = Variable(_merge)
        m1 = F.relu(self.merged(h0))
        
        q_value = self.out(m1)

        return q_value
    
    
    def arr_to_gpu(self, arr):
        return arr if not self.on_gpu else cuda.to_gpu(arr)
    
        
    def show_convolutions(self, big_array):
        # -- show the convolutions
        # big_array : is of dtype=object and is filled with Variable type
        
        # convert into dtype=float filled with float type in order to show the image
        h1mod = np.asarray(big_array)
        for xi in range(len(h1mod)):
            for k in range(len(h1mod[xi])):
                for j in range(len(h1mod[xi][k])):
                    for i in range(len(h1mod[xi][k][j])) :
                        ad = h1mod[xi][k][j]
                        advalue = ad[i].array
                        h1mod[xi][k][j][i] = np.float32(advalue.item())
        h1float = np.ndarray(shape=(len(h1mod),len(h1mod[0]), len(h1mod[0][0]),len(h1mod[0][0][0])), dtype=float)
        # prepare graph data
        for xi in range(len(h1mod)):
            for k in range(len(h1mod[xi])):
                for j in range(len(h1mod[xi][k])):
                    for i in range(len(h1mod[xi][k][j])) :
                        ad = h1mod[xi][k][j]
                        h1float[xi][k][j][i] = np.float(ad[i])
        # prepare graph configs
        fig = plt.figure()
        rows=7
        columns=len(h1mod[xi])/7+1
        xi=0
        for k in range(len(h1mod[xi])):
            if k+1<rows*columns:
                fig.add_subplot(rows,columns,k+1)
                plt.imshow(h1float[xi][k], interpolation='nearest', cmap='gray')
        titless = plt.title('convolution of size '+str(len(h1mod[0][0]))+"x"+str(len(h1mod[0][0][j])))
        # show graph
        plt.show()


class CriticAgent(Agent):
    # the one who decides which action to make
    
    def __init__(self, actions, epsilon=0.1, on_gpu=False, model_path="", load_if_exist=True):
        # -- initialize the agent
        # actions : the possible actions
        # epsilon : the bigger the more we do random actions
        # on_gpu : use gpu specific functions
        # model_path : where the training results are
        # load_if_exist : load the training form the path
        
        # memorize settings
        self.actions = actions
        self.epsilon = epsilon
        self.loss_values =[]
        self.on_gpu = on_gpu
        if on_gpu :
            self.q = Q(1, len(actions), on_gpu).to_gpu()
            self.q_critic = QCritic(1, len(actions), on_gpu).to_gpu()
        else:
            self.q = Q(1, len(actions), on_gpu)
            self.q_critic = QCritic(1, len(actions), on_gpu)
        self._state = []
        self._observations = [ np.zeros((self.q.sizex, self.q.sizey), np.float32), 
                                np.zeros((self.q.sizex, self.q.sizey), np.float32) ]
        self.last_action = 0
        self.last_state=[]
        
        # load training
        self.model_path = model_path if model_path else os.path.join(os.path.dirname(__file__), "./store")
        if not os.path.exists(self.model_path):
            print("make directory to store model at {0}".format(self.model_path))
            os.mkdir(self.model_path)
        else:
            models = self.get_model_files()
            if load_if_exist and len(models) > 0:
                print("load model file {0}.".format(models[-1]))
                serializers.load_npz(os.path.join(self.model_path, models[-1]), self.q)  # use latest model
    
    
    def _update_state(self, observation):
        # -- get latest state in the right format and manage the _state attribute
        # observation : image we want to reformat
        
        formatted = self._format(observation)
        state = np.maximum(formatted, self._observations[0])
        self._state.append(state)
        if len(self._state) > self.q.n_channels:
            self._state.pop(0)
        return formatted
    
    
    @classmethod
    def _format(cls, image):
        # format the given image
        # image : matrix to reformat
        
        im = image[0]
        return im.astype(np.float32)

    
    def start(self, observation, repeat=1):
        # reset attributes for the new episode and return best action
        # observation : first screenshot of our new episode
        
        self._state = []
        self._observations = [  np.zeros((self.q.sizex, self.q.sizey), np.float32), 
                                np.zeros((self.q.sizex, self.q.sizey), np.float32) ]
        self.last_action = 0
        action = self.act(observation, 0, repeat=repeat)
        return action
    
    
    def act(self, observation, reward, framefirstorlast=False, repeat=1):
        # -- get best next action and sometimes explore random actions
        # observation : image from which we will decide what the next action should be
        # reward : here useless but needed for trainer (they are of the same type)
        # framefirstorlast : only used to decide when to show the convolutions in case we chose to 
        #                                (in that case we show them during the end of an episode)
        
        # get important data
        o = self._update_state(observation)
        s = self.get_state()
        if repeat>1:
            big_array=np.zeros((repeat,1,len(observation[0]), len(observation[0][0])), dtype=np.float32)
            big_array[0][0]=s[0]
        else:
            big_array=np.array([s])
        qv = self.q(big_array, framefirstorlast)
        # decide to explore or not
        if np.random.rand() < self.epsilon or math.isnan(qv.data[0][1]):
            action = [np.float32(np.random.randint(0, 1000)/1000), np.float32(np.random.randint(0, 100)/100)] # random action
            b = (action[1]>0.5)
            action[1]=b
        else:
            #print("qv", qv)
            #print(qv.data[0][1])
            #print(qv.data[0][1]>0.5)
            b = (qv.data[0][1]>0.5)
            action = []
            action.append(qv.data[0][0])
            action.append(b)
            #action = np.argmax(qv.data[-1]) # chose an action that fits best the situation described by the observation
        # update data
        self._observations[-1] = self._observations[0].copy()
        self._observations[0] = o
        self.last_action = action
        
        # make it shoot left and right (between -1 and 1)
        return action
    

    def get_state(self):
        state = []
        for  i in range(self.q.n_channels):
            if i < len(self._state):
                state.append(self._state[i])
            else:
                state.append(np.zeros((self.q.sizex, self.q.sizey), dtype=np.float32))
        np_state = np.array(state)
        
        return np_state
    
    
    def save(self, index=0):
        fname = "doom.model" if index == 0 else "doom_{0}.model".format(index)
        path = os.path.join(self.model_path, fname)
        serializers.save_npz(path, self.q)
    
    
    def get_model_files(self):
        files = os.listdir(self.model_path)
        model_files = []
        for f in files:
            if f.startswith("doom") and f.endswith(".model"):
                model_files.append(f)
        
        model_files.sort()
        return model_files

