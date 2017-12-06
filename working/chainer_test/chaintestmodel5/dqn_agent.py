# modified https://github.com/icoxfog417/chainer_pong/blob/master/model/dqn_agent.py
import os
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

class Q(Chain):
    """
    You want to optimize this function to determine the action from state (state is represented by CNN vector)
    """

    sizex = 80  # 80 X y image
    sizey = 80  # x X 80 image
    
    def __init__(self, n_history, n_action, on_gpu=False):
        self.n_history = n_history
        self.n_action = n_action
        self.on_gpu = on_gpu
        super(Q, self).__init__(
            l1=L.Convolution2D(n_history, 32, ksize=8, stride=4, nobias=False, initialW=I.HeNormal(np.sqrt(2) / np.sqrt(2))),
            l2=L.Convolution2D(32, 64, ksize=3, stride=2, nobias=False, initialW=I.HeNormal(np.sqrt(2) / np.sqrt(2))),
            l3=L.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, initialW=I.HeNormal(np.sqrt(2)/ np.sqrt(2))),
            l4=L.Linear(3136, 512, initialW=I.HeNormal(np.sqrt(2)/ np.sqrt(2))),
            #lstm = L.LSTM(3136, 3136),
            out=L.Linear(512, self.n_action, initialW=np.zeros((n_action, 512), dtype=np.float32))
        )
        if on_gpu:
            self.to_gpu()

    def __call__(self, state: np.ndarray, show=False):
        if show:
            #print("ZAWARUDO")
            plt.imshow(state[0][0], interpolation='nearest', cmap='gray')
            titless = plt.title('resized frame 80x80')
            #plt.getp(titless)
            plt.show()
        
        _state = self.arr_to_gpu(state)
        s = Variable(_state)
        h1 = F.relu(self.l1(s))
        
        if show:
            self.show_convolutions(h1)
        
        h2 = F.relu(self.l2(h1))
        
        if show:
            self.show_convolutions(h2)
        
        h3 = F.relu(self.l3(h2))
        
        if show:
            self.show_convolutions(h3)
        
        #hlstm = F.relu(self.lstm(h3))
        #h4 = F.relu(self.l4(hlstm))
        h4 = F.relu(self.l4(h3))
        #hlstm = F.relu(self.lstm(h4))
        q_value = self.out(h4)
        return q_value
    
    def arr_to_gpu(self, arr):
        return arr if not self.on_gpu else cuda.to_gpu(arr)
    
    def show_convolution(self, big_array, xi=0, yj=0):
        # the big_array is of dtype=object and is filled with Variable type
        # we need to convert into dtype=float filled with float type in order to show the image
        h1mod = np.asarray(big_array)
        for k in range(len(h1mod[xi])):
            for j in range(len(h1mod[xi][k])):
                for i in range(len(h1mod[xi][k][j])) :
                    ad = h1mod[xi][k][j]
                    advalue = ad[i].array
                    h1mod[xi][k][j][i] = np.float32(advalue.item())
        h1float = np.ndarray(shape=(len(h1mod[xi][yj]),len(h1mod[xi][yj][j])), dtype=float)
        for j in range(len(h1mod[xi][yj])):
            for i in range(len(h1mod[xi][yj][j])) :
                ad = h1mod[xi][yj][j]
                h1float[j][i] = np.float(ad[i])
        plt.imshow(h1float, interpolation='nearest')
        titless = plt.title('convolution of size '+str(len(h1mod[xi][yj]))+"x"+str(len(h1mod[xi][yj][j])))
        #plt.getp(titless)
        plt.show()
        
    def show_convolutions(self, big_array):
            # the big_array is of dtype=object and is filled with Variable type
            # we need to convert into dtype=float filled with float type in order to show the image
            h1mod = np.asarray(big_array)
            for xi in range(len(h1mod)):
                for k in range(len(h1mod[xi])):
                    for j in range(len(h1mod[xi][k])):
                        for i in range(len(h1mod[xi][k][j])) :
                            ad = h1mod[xi][k][j]
                            advalue = ad[i].array
                            h1mod[xi][k][j][i] = np.float32(advalue.item())
            h1float = np.ndarray(shape=(len(h1mod),len(h1mod[0]), len(h1mod[0][0]),len(h1mod[0][0][0])), dtype=float)
            for xi in range(len(h1mod)):
                for k in range(len(h1mod[xi])):
                    for j in range(len(h1mod[xi][k])):
                        for i in range(len(h1mod[xi][k][j])) :
                            ad = h1mod[xi][k][j]
                            h1float[xi][k][j][i] = np.float(ad[i])
                            
            fig = plt.figure()
            rows=7
            columns=len(h1mod[xi])/7+1
            #for xi in range(len(h1mod)):
            xi=0
            for k in range(len(h1mod[xi])):
                if k+1<rows*columns:
                    fig.add_subplot(rows,columns,k+1)
                    plt.imshow(h1float[xi][k], interpolation='nearest', cmap='gray')
            titless = plt.title('convolution of size '+str(len(h1mod[0][0]))+"x"+str(len(h1mod[0][0][j])))
            #plt.getp(titless)
            plt.show()


class DQNAgent(Agent):
    
    def __init__(self, actions, epsilon=1, n_history=4, on_gpu=False, model_path="", load_if_exist=True):
        self.actions = actions
        self.epsilon = epsilon
        self.q = Q(n_history, len(actions), on_gpu)
        self._state = []
        self._observations = [
            np.zeros((self.q.sizex, self.q.sizey), np.float32), 
            np.zeros((self.q.sizex, self.q.sizey), np.float32)
        ]  # now & pre
        self.last_action = 0
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
        formatted = self._format(observation)
        state = np.maximum(formatted, self._observations[0])
        self._state.append(state)
        if len(self._state) > self.q.n_history:
            self._state.pop(0)
        return formatted
    
    @classmethod
    def _format(cls, image):
        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        #im = resize(rgb2gray(image), (80, 80))
        im = image[0]
        #print(im)
        #plt.imshow(im, interpolation='nearest')
        #plt.show()
        return im.astype(np.float32)

    def start(self, observation):
        self._state = []
        self._observations = [
            np.zeros((self.q.sizex, self.q.sizey), np.float32), 
            np.zeros((self.q.sizex, self.q.sizey), np.float32)
        ]
        self.last_action = 0

        action = self.act(observation, 0)
        return action
    
    def act(self, observation, reward, framefirstorlast=False):
        o = self._update_state(observation)
        s = self.get_state()
        #TODO : show first and last
        qv = self.q(np.array([s]), framefirstorlast) # batch size = 1

        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, len(self.actions))
        else:
            action = np.argmax(qv.data[-1])
        
        self._observations[-1] = self._observations[0].copy()
        self._observations[0] = o
        self.last_action = action

        return action

    def get_state(self):
        state = []
        for  i in range(self.q.n_history):
            if i < len(self._state):
                state.append(self._state[i])
            else:
                state.append(np.zeros((self.q.sizex, self.q.sizey), dtype=np.float32))
        
        np_state = np.array(state)  # n_history x (width x height)
        return np_state
    
    #TODO : change file names to doom names
    def save(self, index=0):
        fname = "pong.model" if index == 0 else "pong_{0}.model".format(index)
        path = os.path.join(self.model_path, fname)
        serializers.save_npz(path, self.q)
    
    def get_model_files(self):
        files = os.listdir(self.model_path)
        model_files = []
        for f in files:
            if f.startswith("pong") and f.endswith(".model"):
                model_files.append(f)
        
        model_files.sort()
        return model_files