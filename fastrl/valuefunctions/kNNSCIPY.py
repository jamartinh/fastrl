from scipy.spatial import KDTree
import numpy as np
from fastrl.valuefunctions.FAInterface import FARL


class kNNQ(FARL):
    def __init__(self, nactions, low, high, n_elemns, k=1, alpha=0.3, lm=0.95):

        self.lbounds = low
        self.ubounds = high
        self.cl = self.ndlinspace(low, high, n_elemns)

        self.k = k
        self.shape = self.cl.shape
        self.nactions = nactions
        self.Q = np.zeros((self.shape[0], nactions)) + 0.0

        # self.Q         = uniform(-100,0,(self.shape[0],nactions))+0.0
        self.e = np.zeros((self.shape[0], nactions)) + 0.0
        # self.ac         = zeros((self.shape[0]))+0.0 #classifiers activation
        self.ac = []
        self.knn = []
        self.alpha = alpha
        self.lm = lm  # good 0.95
        self.last_state = np.zeros((1, self.shape[1])) + 0.0

        self.lbounds = np.array(self.lbounds)
        self.ubounds = np.array(self.ubounds)
        self.cl = np.array(self.RescaleInputs(self.cl))
        self.knntree = KDTree(self.cl, 100)


    def actualize(self):
        self.knntree = KDTree(self.cl, 100)

    def ndlinspace(self, low, high, nelems):
        """ ndlinspace: n-dimensional linspace function
            input_ranges = [[-1,1],[-0.07,0.07]]
            nelems = (5,5)
        """
        x = np.indices(nelems).T.reshape(-1, len(nelems)) + 1.0

        from_b = np.array(nelems, np.float32)

        y = self.lbounds + (((x - 1) / (from_b - 1)) * (self.ubounds - self.lbounds))

        return y

    def Load(self, strfilename):
        self.Q = np.load(strfilename)

    def Save(self, strfilename):
        np.save(strfilename, self.Q)

    def ResetTraces(self):
        self.e *= 0.0
        # self.actualize()

    def RescaleInputs(self, s):
        return self.ScaleValue(np.array(s), self.lbounds, self.ubounds, -1.0, 1.0)

    def ScaleValue(self, x, from_a, from_b, to_a, to_b):
        return to_a + (((x - from_a) / (from_b - from_a)) * (to_b - to_a))

    def GetkNNSet(self, s):

        if np.allclose(s, self.last_state) and self.knn != []:
            return self.knn

        self.last_state = s
        state = self.RescaleInputs(s)

        d, self.knn = self.knntree.query(state, self.k, eps=0.0, p=2)

        print(d.shape)
        print(self.knn.shape)


        self.ac = 1.0 / (1.0 + d**2)  # calculate the degree of activation
        self.ac /= sum(self.ac)
        return self.knn

    def CalckNNQValues(self, M):
        Qvalues = np.dot(np.transpose(self.Q[M]), self.ac)
        return Qvalues

    def get_value(self, s, a=None):
        """ Return the Q value of state (s) for action (a)
        """
        M = self.GetkNNSet(s)

        if a is None:
            return self.CalckNNQValues(M)

        return self.CalckNNQValues(M)[a]

    def update(self, s, a, v, gamma=1.0):
        """ update action value for action(a)
        """

        M = self.GetkNNSet(s)

        if self.lm > 0:
            # cumulating traces
            # self.e[M,a] = self.e[M,a] +  self.ac[M].flatten()

            # replacing traces
            self.e[M] = 0
            self.e[M, a] = self.ac

            td_error = v - self.get_value(s, a)
            self.Q += self.alpha * td_error * self.e
            self.e *= self.lm
        else:
            td_error = v - self.get_value(s, a)
            self.Q[M, a] += self.alpha * td_error * self.ac

            # self.cl[M]+= 0.00005 * ( self.rescale_inputs(s)-self.cl[M] )

    def has_population(self):
        return True

    def get_population(self):
        pop = self.ScaleValue(self.cl, -1.0, 1.0, self.lbounds, self.ubounds)
        for i in range(self.shape[0]):
            yield pop[i]
