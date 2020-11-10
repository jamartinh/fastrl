import numpy as np
from fastrl.valuefunctions.FAInterface import FARL
import faiss
from scipy.special import softmax


class kNNQFaissExt(FARL):

    def __init__(self, nactions, low, high, n_elemns, k=1, alpha=0.3, lm=0.95):

        self.dimension = int(low.shape[0])
        self.lbounds = low
        self.ubounds = high

        self.cl = self.random_space(npoints=280000).astype('float32')

        self.k = k
        self.shape = self.cl.shape
        self.nactions = nactions

        self.Q = np.zeros((self.cl.shape[0], nactions)) + -100.0
        # self.Q         = uniform(-100,0,(self.shape[0],nactions))+0.0

        self.e = np.zeros((self.cl.shape[0], nactions)) + 0.0

        # self.ac         = zeros((self.shape[0]))+0.0 #classifiers activation
        self.ac = []

        self.knn = []
        self.alpha = alpha
        self.lm = lm  # good 0.95
        self.last_state = np.zeros((1, self.shape[1])) + 0.0

        self.lbounds = np.array(self.lbounds)
        self.ubounds = np.array(self.ubounds)

        self.cl_idx = np.array(self.rescale_inputs(self.cl))

        print("building value function memory")
        # self.index = faiss.IndexFlatL2(self.dimension)
        # self.index.add(x=self.cl)

        nlist = 100
        quantizer = faiss.IndexFlatL2(self.dimension)  # the other index
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        assert not self.index.is_trained
        self.index.train(self.cl)
        assert self.index.is_trained
        self.index.add(self.cl)
        self.index.nprobe = 10
        print("value function memory done...")

    def actualize(self):
        self.index.add(x=self.cl)

    def ndlinspace(self, nelems):

        x = np.indices(nelems).T.reshape(-1, len(nelems)) + 1.0

        from_b = np.array(nelems, np.float32)

        y = self.lbounds + (((x - 1) / (from_b - 1)) * (self.ubounds - self.lbounds))

        return y

    def random_space(self, npoints):
        d = []
        for l, h in zip(self.lbounds, self.ubounds):
            d.append(np.random.uniform(l, h, (npoints, 1)))

        return np.concatenate(d, 1)

    def load(self, str_filename):
        self.Q = np.load(str_filename)

    def save(self, str_filename):
        np.save(str_filename, self.Q)

    def reset_traces(self):
        self.e *= 0.0
        # self.actualize()

    def rescale_inputs(self, s):
        return self.scale_value(np.array(s), self.lbounds, self.ubounds, -1.0, 1.0)

    def scale_value(self, x, from_a, from_b, to_a, to_b):
        return to_a + (((x - from_a) / (from_b - from_a)) * (to_b - to_a))

    def get_knn_set(self, s):

        if self.last_state is not None:
            if np.allclose(s, self.last_state,rtol=1e-03, atol=1e-04) and self.knn != []:
            # if np.allclose(s, self.last_state) and self.knn != []:
                return self.knn

        self.last_state = s
        state = self.rescale_inputs(s)

        d, self.knn = self.index.search(x=np.array([state]).astype(np.float32), k=self.k)
        d = np.squeeze(d)

        # if self.index.ntotal < self.max_points and self.Q[self.knn, :].flatten().std() > 0.1:
        #     # print(d[0])
        #     # print(self.Q[self.knn,:].flatten().std())
        #     self.index.add(x=np.array(np.array([state])))
        #     d, self.knn = self.index.search(x=np.array([state]).astype(np.float32), k=self.k)
        #     d = np.squeeze(d)

        self.knn = np.squeeze(self.knn)

        self.ac = 1.0 / (1.0 + d)  # calculate the degree of activation
        self.ac /= sum(self.ac)
        #self.ac = softmax(-np.sqrt(d))

        return self.knn

    def calculate_knn_q_values(self, M):
        Q_values = np.dot(np.transpose(self.Q[M]), self.ac)
        return Q_values

    def get_value(self, s, a=None):
        """ Return the Q value of state (s) for action (a)
        """
        M = self.get_knn_set(s)

        if a is None:
            return self.calculate_knn_q_values(M)

        return self.calculate_knn_q_values(M)[a]

    def update(self, s, a, v, gamma=1.0):
        """ update action value for action(a)
        """

        M = self.get_knn_set(s)

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
        pop = self.scale_value(self.cl, -1.0, 1.0, self.lbounds, self.ubounds)
        for i in range(self.shape[0]):
            yield pop[i]
