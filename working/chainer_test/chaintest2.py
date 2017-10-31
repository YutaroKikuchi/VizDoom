"""
Very simple implementation for MNIST training code with Chainer using
Multi Layer Perceptron (MLP) model
 
This code is to explain the basic of training procedure.
 
"""
from __future__ import print_function
import time
import os
import numpy as np
import six
import matplotlib.pyplot as plt
 
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import serializers
from PIL import Image
 
 
class MLP(chainer.Chain):
    """Neural Network definition, Multi Layer Perceptron"""
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__(
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
 
 
class SoftmaxClassifier(chainer.Chain):
    """Classifier is for calculating loss, from predictor's output.
    predictor is a model that predicts the probability of each label.
    """
    def __init__(self, predictor):
        super(SoftmaxClassifier, self).__init__(
            predictor=predictor
        )
 
    def __call__(self, x, t):
        y = self.predictor(x)
        self.loss = F.softmax_cross_entropy(y, t)
        self.accuracy = F.accuracy(y, t)
        return self.loss
 
 
def main():
    # Configuration setting
    gpu = -1                  # GPU ID to be used for calculation. -1 indicates to use only CPU.
    batchsize = 100           # Minibatch size for training
    epoch = 20                # Number of training epoch
    out = 'result/1_minimum'  # Directory to save the results
    unit = 50                 # Number of hidden layer units, try incresing this value and see if how accuracy changes.
 
    print('GPU: {}'.format(gpu))
    print('# unit: {}'.format(unit))
    print('# Minibatch-size: {}'.format(batchsize))
    print('# epoch: {}'.format(epoch))
    print('out directory: {}'.format(out))
 
    # Set up a neural network to train
    model = MLP(unit, 10)
    # Classifier will calculate classification loss, based on the output of model
    classifier_model = SoftmaxClassifier(model)
 
    if gpu >= 0:
        chainer.cuda.get_device(gpu).use()  # Make a specified GPU current
        classifier_model.to_gpu()           # Copy the model to the GPU
    xp = np if gpu < 0 else cuda.cupy
 
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(classifier_model)
 
    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()
    print("-info-")
    print(len(train))
    print(len(train[0]))
    print(type(train[0][0]))
    print(train[0][0])
    print(type(train[0][1]))
    print(train[0][1])
    
    slice56 = np.random.random((28*28, 28*28))
    print(len(slice56))
    print(len(train[0][0]))
    for i in range(len(train[0][0])):
        # print(i)
        slice56[i] = train[0][0][i]
    #slice56 = train[0][0]
    # convert values to 0 - 255 int8 format
    formatted = (slice56 * 255).astype('uint8')
    print(formatted)
    img = Image.fromarray(formatted)
    img.show()
    print("------------")
 
    n_epoch = epoch
    N = len(train)       # training data size
    N_test = len(test)  # test data size
 
    # Learning loop
    for epoch in range(1, n_epoch + 1):
        print('epoch', epoch)
 
        # training
        perm = np.random.permutation(N)
        sum_accuracy = 0
        sum_loss = 0
        start = time.time()
        for i in six.moves.range(0, N, batchsize):
            x = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][0]))
            t = chainer.Variable(xp.asarray(train[perm[i:i + batchsize]][1]))
 
            # Pass the loss function (Classifier defines it) and its arguments
            optimizer.update(classifier_model, x, t)
 
            sum_loss += float(classifier_model.loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)
        end = time.time()
        elapsed_time = end - start
        throughput = N / elapsed_time
        print('train mean loss={}, accuracy={}, throughput={} images/sec'.format(
            sum_loss / N, sum_accuracy / N, throughput))
 
        # evaluation
        sum_accuracy = 0
        sum_loss = 0
        for i in six.moves.range(0, N_test, batchsize):
            index = np.asarray(list(range(i, i + batchsize)))
            x = chainer.Variable(xp.asarray(test[index][0]))
            t = chainer.Variable(xp.asarray(test[index][1]))
 
            loss = classifier_model(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(classifier_model.accuracy.data) * len(t.data)
 
        print('test  mean loss={}, accuracy={}'.format(
            sum_loss / N_test, sum_accuracy / N_test))
 
    # Save the model and the optimizer
    if not os.path.exists(out):
        os.makedirs(out)
    print('save the model')
    serializers.save_npz('{}/classifier_mlp.model'.format(out), classifier_model)
    serializers.save_npz('{}/mlp.model'.format(out), model)
    print('save the optimizer')
    serializers.save_npz('{}/mlp.state'.format(out), optimizer)
    
    print("----begin end------")
    
    #print(flat_arr.astype(np.float32)/255)
    #print(flat_arr.astype(np.float32)[0])
    #print(flat_arr.astype(np.float32)[1])
    #print(len(flat_arr.astype(np.float32)))
    #print("test from database mnist")
    #x = chainer.Variable(xp.asarray(train[0][0]))
    #print(classifier_model(x))
     
    
    def predict(model, x_test):
        x = chainer.Variable(x_test)

        #h1 = F.dropout(F.relu(model.predictor.l1(x)))
        #h2 = F.dropout(F.relu(model.predictor.l2(h1)))
        #y = model.predictor.l3(h2)
        y = model.predictor(x)
        print(y.data)
        return np.argmax(y.data)

    modelUsed = L.Classifier(model)
    
    # testing 3 images, they are inverted which means that the backgroud is black and ink white
    # it doesn t seem to work when it is the other way around (probably because of the training images, didn t check)
    
    print("should be one - invert") # works, comes from database mnist
    img = Image.open('invert_one.png').convert('L') #.convert('RGBA')
    arr = np.array(img)
    flat_arr = arr.ravel()
    toscan = flat_arr.astype(np.float32)/255
    print(type(toscan))
    print(type(toscan[0]))
    print(type(train[0][0]))
    print(type(train[0][0][0]))
    x = chainer.Variable(toscan)
    toscan= toscan.reshape(28* 28)
    toscanUsed = np.array([toscan], dtype=np.float32)
    print(predict(modelUsed, toscanUsed))
    print(predict(modelUsed, toscanUsed))
    
    print("should be eight - invert") # works, comes from database mnist
    img = Image.open('invert_eight.png').convert('L') #.convert('RGBA')
    arr = np.array(img)
    flat_arr = arr.ravel()
    toscan = flat_arr.astype(np.float32)/255
    print(type(toscan))
    print(type(toscan[0]))
    print(type(train[0][0]))
    print(type(train[0][0][0]))
    x = chainer.Variable(toscan)
    toscan= toscan.reshape(28* 28)
    toscanUsed = np.array([toscan], dtype=np.float32)
    print(predict(modelUsed, toscanUsed))
    print(predict(modelUsed, toscanUsed))
    
    print("should be six - invert") # doesn t really work because of pencil thickness that is different
    img = Image.open('invert_six.png').convert('L') #.convert('RGBA')
    arr = np.array(img)
    flat_arr = arr.ravel()
    toscan = flat_arr.astype(np.float32)/255
    print(type(toscan))
    print(type(toscan[0]))
    print(type(train[0][0]))
    print(type(train[0][0][0]))
    x = chainer.Variable(toscan)
    toscan= toscan.reshape(28* 28)
    toscanUsed = np.array([toscan], dtype=np.float32)
    print(predict(modelUsed, toscanUsed))
    print(predict(modelUsed, toscanUsed))
    
    print("should be six - c invert") # custom number with right pencil thickness - ok
    img = Image.open('c_sixc.png').convert('L') #.convert('RGBA')
    arr = np.array(img)
    flat_arr = arr.ravel()
    toscan = flat_arr.astype(np.float32)/255
    print(type(toscan))
    print(type(toscan[0]))
    print(type(train[0][0]))
    print(type(train[0][0][0]))
    x = chainer.Variable(toscan)
    toscan= toscan.reshape(28* 28)
    toscanUsed = np.array([toscan], dtype=np.float32)
    print(predict(modelUsed, toscanUsed))
    print(predict(modelUsed, toscanUsed))
    
    print("should be seven - c invert") # custom number with right pencil thickness - doesn t work, looks like a 2
    arr = np.array(img)
    flat_arr = arr.ravel()
    toscan = flat_arr.astype(np.float32)/255
    print(type(toscan))
    print(type(toscan[0]))
    print(type(train[0][0]))
    print(type(train[0][0][0]))
    x = chainer.Variable(toscan)
    toscan= toscan.reshape(28* 28)
    toscanUsed = np.array([toscan], dtype=np.float32)
    print(predict(modelUsed, toscanUsed))
    print(predict(modelUsed, toscanUsed))
    
    print("should be one - c invert") # custom number with right pencil thickness - ok
    img = Image.open('c_onec.png').convert('L') #.convert('RGBA')
    arr = np.array(img)
    flat_arr = arr.ravel()
    toscan = flat_arr.astype(np.float32)/255
    print(type(toscan))
    print(type(toscan[0]))
    print(type(train[0][0]))
    print(type(train[0][0][0]))
    x = chainer.Variable(toscan)
    toscan= toscan.reshape(28* 28)
    toscanUsed = np.array([toscan], dtype=np.float32)
    print(predict(modelUsed, toscanUsed))
    print(predict(modelUsed, toscanUsed))
    
    print("should be three - c invert") # custom number with right pencil thickness - ok
    img = Image.open('c_threec.png').convert('L') #.convert('RGBA')
    arr = np.array(img)
    flat_arr = arr.ravel()
    toscan = flat_arr.astype(np.float32)/255
    print(type(toscan))
    print(type(toscan[0]))
    print(type(train[0][0]))
    print(type(train[0][0][0]))
    x = chainer.Variable(toscan)
    toscan= toscan.reshape(28* 28)
    toscanUsed = np.array([toscan], dtype=np.float32)
    print(predict(modelUsed, toscanUsed))
    print(predict(modelUsed, toscanUsed))

    
    #print(classifier_model(x))
    print("-------------------")
 
if __name__ == '__main__':
    main()