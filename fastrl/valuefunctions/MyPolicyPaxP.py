#-------------------------------------------------------------------------------
# Copyright (c) 2012 Jose Antonio Martin H..
# All rights reserved. This program and the accompanying materials
# are made available under the terms of the GNU Public License v3.0
# which accompanies this distribution, and is available at
# http://www.gnu.org/licenses/gpl.html
#
# Contributors:
#     Jose Antonio Martin H. - Translation to Python from Java
#-------------------------------------------------------------------------------
#package myPolicy;
from numpy import *
from numpy.random import seed, uniform, rand, normal

import random
import math
import datetime
from collections import defaultdict, deque


from exploChallenge.policies.ContextualBanditPolicy import ContextualBanditPolicy
from exploChallenge.logs.yahoo.YahooArticle import YahooArticle
from exploChallenge.logs.yahoo.YahooVisitor import YahooVisitor

class MyPolicyPaxP(ContextualBanditPolicy):

    def __init__(self):
        #Any initialization of your algorithm should be done here.
        self.N = 136 # number of features
        self.selections = ones((700, self.N) , dtype = int)
        self.selectionst = ones((700, 13, 32) , dtype = int)
        self.P = ones((700, self.N) , dtype = float)
        self.Pday = ones((700, 13, 32) , dtype = float)
        self.acode = defaultdict(lambda: len(self.acode) + 2)


        self.epsilon = 0.00001
        #self.sigma = 3


    #@Override
    def getActionToPerform(self, visitor, possibleActions):
        # Given a visitor, you have to choose the "best" article in the list.

        self.aset = [self.acode[a.yid] for a in possibleActions]

        #self.timestamp = visitor.timestamp

        visitor.features[0] = 1
        self.i = nonzero(visitor.features)[0]
        self.d = datetime.datetime.fromtimestamp(visitor.timestamp).day
        self.m = datetime.datetime.fromtimestamp(visitor.timestamp).month

        if random.random() <= self.epsilon:
            TOP5 = possibleActions[-5:]
            return random.choice(possibleActions + TOP5 + TOP5)
#            p = self.preferences(self.aset)
#            acts = argsort(-p)#[0:5]
#            #best = random.choice(acts)
#            n = self.normal_choice(len(possibleActions))
#            best = acts[n]
#            return possibleActions[best]


        p = self.preferences(self.aset)
        best = p.argmax()
        return possibleActions[best]

    def normal_choice(self, maxlen = 30):
        i = int(math.floor(abs(random.gauss(0, self.sigma))))
        return min(i, maxlen - 1)


    def preferences(self, a):

        d = self.d
        m = self.m
        i = self.i
        factor = float(self.N) - i.size

        Pday = self.Pday[a, m, d]# / amin(self.Pday[a, m, :], axis = 1)  #P(x|a)
        Px_a = self.P[a, :]  #P(x|a)        
        iPx = self.P[1, i]    #P(x)

        M = (Px_a[:, i] * iPx)
        A = (factor + sum(M , axis = 1)) / (sum(iPx) + factor)
        B = Pday
        C = prod(Px_a[:, i]  , axis = 1)

        return A * B * C

    #@Override
    def updatePolicy(self, visitor, action, reward):
        #self.history.append((visitor, self.possibleActions, action, reward))
        # update your policy given the visitor, the displayed article and
        # the associated reward (click or not click)

        a = self.acode[action.yid]
        m = self.m
        d = self.d
        i = self.i

        self.selections[a, i] = minimum(self.selections[a, i] + 1, 2000)
        self.selections[0, i] = minimum(self.selections[0, i] + 1, 2000)
        self.selectionst[a, m, d] = minimum(self.selectionst[a, m, d] + 1, 2000)

        reward = max(1E-15, reward) * 1.25

        self.P[a, i] += (reward - self.P[a, i]) / self.selections[a, i]
        self.P[0, i] += (reward - self.P[0, i]) / self.selections[0, i]
        self.Pday[a, m, d] += (reward - self.Pday[a, m, d]) / self.selectionst[a, m, d]
        self.P[1, :] = 1.0 / self.P[0, :]



#    



