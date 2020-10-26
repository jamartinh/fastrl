from fastrl.valuefunctions.FAInterface import FARL
from numpy import *
from numpy.random import *
from scikits.ann import kdtree


class kNNQ(FARL):

    def __init__(self,nactions,input_ranges,nelemns=[],npoints=0,k=1,alpha=0.3,lm=0.90):
        if  not (nelemns==False)^(npoints==False):
            raise ValueError('Plese indicate either: [nelemns] Xor [npoints]')

        if nelemns:
            #self.cl = self.CreateFullspace(input_ranges,nelemns)
            self.cl = self.ndlinspace(input_ranges,nelemns)
        else:
            self.cl = self.CreateRandomSpace(input_ranges,npoints)

        self.lbounds = []
        self.ubounds = []

        self.k          = k
        self.shape      = self.cl.shape
        self.nactions   = nactions
        self.Q          = zeros((self.shape[0],nactions))
        self.ps         = zeros((self.shape[0],nactions,self.shape[1]))
        #self.Q         = uniform(-1,0,(self.shape[0],nactions))+0.0
        self.e          = zeros((self.shape[0],nactions))+0.0
        #self.ac         = zeros((self.shape[0]))+0.0 #classifiers activation
        self.ac         = []
        self.knn        = []
        self.alpha      = alpha
        self.lm         = lm 
        self.last_state = zeros((1,self.shape[1]))+0.0
        self.next_state = array(self.last_state)
       

        
        
        for r in input_ranges:
            self.lbounds.append(r[0])
            self.ubounds.append(r[1])


        
        self.lbounds = array(self.lbounds)
        self.ubounds = array(self.ubounds)
        self.cl = array (self.RescaleInputs(self.cl))
        self.knntree = kdtree(self.cl)
        
       #self.d2 = add.reduce(self.cl,1)

    def ndtuples(self,*dims):
       """Fast implementation of array(list(ndindex(*dims)))."""

       # Need a list because we will go through it in reverse popping
       # off the size of the last dimension.
       dims = list(dims)

       # N will keep track of the current length of the indices.
       N = dims.pop()

       # At the beginning the current list of indices just ranges over the
       # last dimension.
       cur = arange(N)
       cur = cur[:,newaxis]

       while dims != []:
           d = dims.pop()
           # This repeats the current set of indices d times.
           # e.g. [0,1,2] -> [0,1,2,0,1,2,...,0,1,2]
           cur = kron(ones((d,1)),cur)
           # This ranges over the new dimension and 'stretches' it by N.
           # e.g. [0,1,2] -> [0,0,...,0,1,1,...,1,2,2,...,2]
           front = arange(d).repeat(N)[:,newaxis]
           # This puts these two together.
           cur = column_stack((front,cur))
           N *= d

       return cur
    
    def ndlinspace(self,input_ranges,nelems):
        x = self.ndtuples(*nelems)+1.0        
        lbounds = []
        ubounds = []
        from_b  = array(nelems,float)
        for r in input_ranges:
            lbounds.append(r[0])
            ubounds.append(r[1])            
            
        lbounds = array(lbounds,float)
        ubounds = array(ubounds,float)
        y = (lbounds) + (((x-1)/(from_b-1))*((ubounds)-(lbounds)))        
        return y

        
    def ResetTraces(self):
        self.e = zeros((self.shape[0],self.nactions))+0.0
        
        
    def RescaleInputs(self,s):
        return self.ScaleValue(array(s),self.lbounds,self.ubounds,-1.0,1.0)
    
    def ScaleValue(self,x,from_a,from_b,to_a,to_b):
        return (to_a) + (((x-from_a)/(from_b-from_a))*((to_b)-(to_a)))
    

    def CreateRandomSpace(self,input_ranges,npoints):
        d = []
        x = array([])
        for r in input_ranges:
            d.append( uniform(r[0],r[1],(npoints,1)))


        return concatenate(d,1)

    def CreateFullspace(self,input_ranges,nelems):

        d=[]
        
        for i in range(len(input_ranges)):
            r = input_ranges[i]
            n = nelems[i]
            x = linspace(r[0],r[1],num=n)
            #xdiv = (r[1]-r[0])/float(n)
            #x = arange(r[0],r[1]+xdiv,xdiv)
            d.append(list(x))

        space = d[0]
        for i in range(1,len(d)):
            space = self.crossproduct(space,d[i])

        return array(space)


    def crossproduct(self,a,b):
        ret_list =[]

        for e1 in a:
            for e2 in b:
                if type(e1)!=type([]):
                    x1 = [e1]
                else:
                    x1 = list(e1)

                if type(e2)!=type([]):
                    x2 = [e2]
                else:
                    x2 = list(e2)

                x1+=x2
                ret_list.append(x1)

        return ret_list


    def GetkNNSet(self,s):
        if allclose(s,self.last_state):
             return self.knn

        #s = self.next_state
        self.last_state = s
        state   = self.RescaleInputs(s)
        #fstate  = self.rescale_inputs(self.next_state)
        
        
        (self.knn,self.d) = self.knntree.knn(state, self.k, self.k)#0.1 error good           
        #self.d = sqrt(self.d) #898 MC
        self.ac  = 1.0/(1.0+self.d) # calculate the degree of activation
        self.ac /= sum(self.ac)        
        return self.knn
    

    def CalckNNQValues(self,M):
        Qvalues = dot(transpose(self.Q[M]),self.ac)
        return Qvalues

    

    def get_value(self, s, a=None):
        """ Return the Q value of state (s) for action (a)
        """        
        M = self.GetkNNSet(s)
            
        if a==None:
            return self.CalckNNQValues(M)
            

        return self.CalckNNQValues(M)[a]

    def FilterState(self,s,a,sp):
        M  = self.GetkNNSet(s)
        Predicted_sp = dot(transpose(self.ps[M,a]),self.ac)
        delta_sp = sp-Predicted_sp
        self.ps[M,a] += 0.1*reshape(self.ac,(self.k,1)) * delta_sp

        new_sp = dot(transpose(self.ps[M,a]),self.ac)
        return new_sp

    def update(self, s, a, v, gamma=1.0):
        """ update action value for action(a)
        """
        M = self.GetkNNSet(s)
        
        #cumulating traces
        #self.e[M,a] = self.e[M,a] +  self.ac

        #replacing traces
        self.e[M]   =  0.0
        self.e[M,a] =  self.ac
        TD_error    =  v - self.get_value(s, a)
        self.Q +=  self.alpha * (TD_error) * self.e        
        #add(self.Q, self.alpha * (TD_error) * self.e,self.Q)
        self.e *= self.lm

       
        
        
    def has_population(self):
        return True

    def get_population(self):
        pop = self.ScaleValue(self.cl,-1.0,1.0,self.lbounds,self.ubounds)
        for i in range(self.shape[0]):
            yield pop[i]


    
        

