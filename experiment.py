#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:28:22 2017

#   In this senario, both bandits are always played: the agent chooses one, while another "guest" plays another
#   the agent is only perceives the "jackpot bell" from the non-played bandit and is unaware of the bandit reward

@author: remi
"""
import pandas as pd
import math as mt
import numpy as np
import timeit
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import bandit as bd
import pgnet as pg

#Training function
#   Trains an agent network [net] for a specified amount of time [trainTime]
#   Training input is rolling history of past actions, of depth [histDepth]
#   Episodes are completed when either 
#   - the maximum number of steps is reached. [epLen]
#   - the agent has achieved a target score. [epTarget]
#   Verbosity: prints reward every [verbose] episode.
def resumeTrain(net,histDepth, trainTime, epLen,epTarget,verbose=0):

    #Check parameters
    if epLen==0 and epTarget==0:
        print('Define training length or target!')

    #Global game parameters
    cp=1    #Cash played for every episode
    histDepth=int(histDepth)
    
    #Start learning
    rewards = []
    rewardAvg = []

    ep=0
    start_time = timeit.default_timer()
    while(timeit.default_timer() - start_time) < trainTime:
        ep+=1
        epDur=0
        gameHist=np.zeros(histDepth)

        while (epDur<epLen or epLen==0) and (abs(sum(rewards))<epTarget or epTarget==0):
            epDur+=1

            #Decision: decide on which machine to play, based on the rolling history
            action = net.propForward(gameHist)[0]

            #Game retults,  append to the game history
            #gameRes=[bandit1.play(),bandit2.play()]
            gameRes=cas.play(cp)

            # recod reward: gain on the machine selected by action - investment
            rewards.append(gameRes[action]-cp)

            # Append to the game signal to history: obfuscate gains (we only hear the "jackpot bell")
            gameHist = np.append(gameHist[2:], np.array(gameRes)>0)
            
        #Verbose
        if verbose>0 and ep%verbose==0:
            print('Reward: ',sum(rewards))
        
        #Collect rewards
        rewardAvg.append(sum(rewards)/len(rewards))

        #Finished episone, now learn
        epReward = np.vstack(rewards)
        rewards = [] # reset array memory

        # compute the discounted reward backwards through time
        epRewardDisc = pg.discount_rewards(epReward)

        #Update network
        net.updateNet(epRewardDisc)

    return rewardAvg



#initalize environment (bandits)
cas=bd.cCasino([5,10],.1)

#Initialize network
histDepth=20
net=pg.cNnet([histDepth,5,1])

#Train
t1=resumeTrain(net,histDepth, 60*3,1000,0,10)

#Plot perf curve
plt.plot(pd.Series(range(0,len(t1)))/len(t1),pd.Series(t1).rolling(center=False,window=100).mean())
