from fastrl.valuefunctions.FAInterface import FARL
from .SOM import SOM
from numpy import *
import numpy.ma as masked

class  SOMQ(FARL):


    def __init__(self,nactions, size_x, size_y,input_ranges,alpha=0.3):

        self.nactions = nactions
        self.alpha = alpha

        self.size_x=size_x
        self.size_y=size_y


        #Create the function approximator
        self.nvars    = len(input_ranges)

        som_ranges = input_ranges + [[0,0] for i in range(nactions)]
        self.Net = SOM(size_I=size_x,size_J=size_y,size_K=self.nvars+self.nactions,input_ranges=som_ranges)


        maskx = zeros((self.nvars+self.nactions))
        #print maskx.shape
        print(self.nvars,self.nvars+self.nactions)
        maskx[self.nvars:self.nvars+self.nactions] = 1
        self.masked_state = masked.array(zeros((self.nvars+self.nactions)),mask=maskx)



    def get_value(self, s, a=None):
        """ Return the Q value of state (s) for action (a)

        """
        self.masked_state[0:self.nvars] = s
        self.Net.Propagate(self.masked_state)
        values = self.Net.W[self.Net.i_min,self.Net.j_min,self.nvars:self.nvars+self.nactions]

        if a==None:
            v = values
            return v


        v = values[a]
        return v


    def update(self, s, a, v):
        """ update action value for action(a)

        """
        self.get_value(s)

        maskx = zeros(self.nvars+self.nactions)
        maskx[self.nvars:self.nvars+self.nactions] = 1
        maskx[self.nvars+a] = 0
        X = masked.array(zeros((self.nvars+self.nactions)),mask=maskx)
        X[0:self.nvars] = s
        X[self.nvars+a] = v
        self.Net.Learn(X)

    def update_all(self, s, v):
        """ update action value for action(a)

        """
        for i in range(self.nactions):
            self.update(s, i, v[i])

    def has_population(self):
        return True

    def get_population(self):
        for i in range(self.size_x):
            for j in range(self.size_y):
                yield self.Net.W[i,j,0:self.nvars]





