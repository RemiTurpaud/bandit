#Policy gradients algorithm based on Andrej Karpathy's http://karpathy.github.io/2016/05/31/rl/

import pandas as pd
import math as mt
import numpy as np

class cNnet:

    def __init__(self, layout):
        # Learning parameters
        self.learning_rate = 1e-4
        self.decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2

        #Structure
        self.initNet(layout)

    def initNet(self,layers):
        self.net = {}
        self.layers = {}

        for l in range(0,len(layers)):
            self.layers[l]=layers[l]

        #Initialize layer weights: starts at 1  as layer 0 is input
        for l in range(1,len(layers)):
            self.net[l] = np.random.randn(self.layers[l], self.layers[l-1]) / np.sqrt(self.layers[l-1])  # "Xavier" initialization

        self.initBuffers()

    def initBuffers(self):
        #Reset buffers
        self.epX, self.epP, self.epA = [], [], []
        self.epH={}
        for l in self.layers.keys():
            self.epH[l]=[]

    def sigmoid(self,x):
      return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

    # forward the policy network and sample an action from the returned probability
    def propForward(self,x):
        #Store the intermediate hidden states to propagate forward
        h=[]
        for l in range(0,len(self.layers)):
            if l==0: #If this is the first layer, then the input is the network input (x)
                h.append(x)
            else:
                h.append(np.dot(self.net[l], h[l-1]))
                #If this is a hidden unit, ReLU
                if l<len(self.layers)-1:
                    h[l][h[l]<0] = 0 # ReLU nonlinearity

            #Store intermediary layer value
            self.epH[l].append(h[l])

        #Output: proba, based on the last layer's output
        p = self.sigmoid(h[l])

        #Determine action
        a = 1 if np.random.uniform() < p else 0  # sample action from distrib

        #Record states and actions
        self.epX.append(x)
        self.epP.append(p)
        self.epA.append(a)

        return a, p # return action and probability of taking action

    #Update netword by consuming the stack of observed and intermediate states
    def updateNet(self, epReward):
        #Transform to arrays
        self.epX = np.vstack(self.epX)
        self.epP = np.vstack(self.epP)
        self.epA = np.vstack(self.epA)

        for l in self.epH.keys():
            self.epH[l]=np.vstack(self.epH[l])

        #Modulates reward by proba to take action
        epReward *= (self.epA-self.epP)

        #Update network by gradient descent
        self.gradDescent(self.propBackward(self.epX, self.epH, epReward))

        #Reset buffers
        self.initBuffers()

    #Propagates backwards and return gradients
    def propBackward(self,epX, epH, epReward):

        grad={}
        dh=epReward

        for l in range(len(self.layers)-1,0,-1):
            #If this is the last layer
            if l == len(self.layers)-1:
                # Determine gradient for last layer: Hidden output*Reward
                grad[l] = np.dot(epReward.T,epH[l-1])
            else:
                # Determine gradient for hidden layer output
                dh = np.dot(dh, self.net[l + 1])  # Back propagate the reward: Determine contribution of current layer
                dh[epH[l] <= 0] = 0  # Rectification: Parametric Rectified Linear Unit
                grad[l] = np.dot(dh.T, epH[l-1])

        return grad

    #Gradient descent with RMSprop
    def gradDescent(self,grad):
        rmsprop_cache = {k: np.zeros_like(v) for k, v in self.net.items()}  # rmsprop memory
        # perform RMSprop parameter update
        for k, v in self.net.items():
            g = grad[k]  # gradient
            rmsprop_cache[k] = self.decay_rate * rmsprop_cache[k] + (1 - self.decay_rate) * g ** 2  # Exp decaying average of squared gradients
            self.net[k] += self.learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)  #

#Discount rewards function
#   r       : reward
#   gamma   : discount factor for reward
#   norm    : normalization function
def discount_rewards(r,gamma=0.99 ,normalize='norm'):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add


    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    if normalize=='norm':
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)

    if normalize=='ratio':
        discounted_r /= len(discounted_r)

    return discounted_r
