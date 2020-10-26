from numpy import *
from numpy.linalg import *
import time
import pickle




def matmult(a,b):
    N = size(a)
    acum = 0
    for i in range(N):
        acum=acum+ a[i]*b[i]
    return acum



class SOM:
    def __init__(self,size_I=10,size_J=20,size_K=10,alpha=0.3,input_ranges=None):
        random.seed(3)
        self.I = size_I
        self.J = size_J
        self.K = size_K
        self.W = zeros((self.I,self.J,self.K))

        if input_ranges==None:
            self.W = random.uniform(-2,2,(self.I,self.J,self.K))
            #self.W = zeros((self.I,self.J,self.K))
        else:
            self.CreateRandomWights(input_ranges,size_I,size_J)


        self.Y = sum(self.W,2)
        self.T = 0.0
        self.Alpha0 = alpha
        self.Alpha  = self.Alpha0
        self.N_Iterations = 1
        self.H = zeros((self.I,self.J,self.K))+0.0
        self.i_min = 0
        self.j_min = 0
        self.best_match = 0
        self.min_val = 9999999999999999
        self.Ratio0 = 0.1
        #self.Ratio0 = max(self.J,self.I)
        self.Ratio  = self.Ratio0


    def CreateRandomSpace(self,input_ranges,npoints):
        d = []
        x = array([])
        for r in input_ranges:
            d.append( random.uniform(r[0],r[1],(npoints,1)))


        return concatenate(d,1)

    def CreateRandomWights(self,input_ranges,size_i,size_j):

        X = self.CreateRandomSpace(input_ranges,size_i*size_j)
        ind = 0
        for i in range(size_i):
            for j in range(size_j):
                self.W[i,j,:] = X[ind]
                ind = ind + 1



    def SaveW(self,filename):
         DATA = [self.W,self.Alpha0,self.Alpha,self.Ratio,self.Ratio0]
         pickle.dump(DATA,open(filename,'w'))


    def LoadW(self,filename):
         DATA = pickle.load(open(filename,'r'))
         [self.W,self.Alpha0,self.Alpha,self.Ratio,self.Ratio0]=DATA


    def mynorm(self,x,ax=2):
        return sqrt(sum(x**2,ax))

    def nsp(self,w,x):
        x.astype(float)
        x = x + 0.0 #it is neccesary to convert x into a float
        return 1 + ( dot(w,x) / (self.mynorm(w)*norm(x)) )


    def alpha(self):
        #Ritmo de Aprendizaje
        self.Alpha = self.Alpha0+(0.01-self.Alpha0) * ( self.T / self.N_Iterations)
        return self.Alpha

    def R(self):
        #Radio de Vecindad
        self.Ratio = self.Ratio0+(1.0-self.Ratio0)*( self.T / self.N_Iterations)
        return self.Ratio


    def dist(self,i,j,k):
        return  sqrt( ((i-self.i_min)**2) + ((j-self.j_min)**2) )


    def H_i_g(self):
        # actualiza el radio de vecindad
        self.H = fromfunction( self.dist , (self.I,self.J,self.K))
        R = self.R()
        #self.H = where (self.H <= R,0.5/(1.0+self.H**2),0.0)
        self.H = where (self.H <= R,1,0.0)


    def NeuroWinner(self):
        #Determinar la Neurona Ganadora
        pos             = argmin(self.Y.flatten())
        self.best_match = self.Y.flatten()[pos]
        self.i_min      = pos / self.J
        self.j_min      = pos % self.J


    def Propagate(self,X):
        #Calcular la Distancia Euclidea = Propagar el estimulo
        self.Y = sum((self.W-X)**2,2)
        #print "Vector X",X.mask
        #print "Valores de Y",self.Y
        # Calcular la inversa del producto escalar normalizado
        #self.Y = 2.0-self.nsp(self.W,X)
        #Determinar la Neurona Ganadora
        self.NeuroWinner()

    def Learn(self,X,reward=1):
        #Actulizar la matriz W de pesos
        self.H_i_g()
        #Actualizar Pesos
        self.W += self.alpha() * self.H * ( X - self.W) * reward
        self.W = array(self.W)

    def Print(self,X):
        print("Entrada:                   " ,X)
        print("La Neurona Ganadora es:    " ,[self.i_min,self.j_min])
        print("Con Vector Caracteristico: " , self.W[self.i_min,self.j_min,])
        #print "Ratio,T,Alpha",self.Ratio,self.T,self.Alpha

    def Train(self,X,N=1000):
        # X es un array de vectores de entrenamiento
        self.N_Iterations=N
        num_samples_vectors=X.shape[0]
        for i in range(N):
            self.T+=1.0
            #print "iteration #:",i
            #print "Ratio,T,Alpha",self.Ratio,self.T,self.Alpha
            for j in range(num_samples_vectors):
                self.Propagate(X[j])
                self.Learn(X[j])


    def ClasifyPattern(self,X):
        self.Propagate(X)
        self.Print(X)



def PruebaSOM():
    red=SOM(5,5,5)
    Entrada=array([[0,0,0,0,0],
                   [64,64,64,64,64],
                   [128,128,128,128,128],
                   [255,255,255,255,255],
                   ])

    t1=time.clock()
    red.Train(Entrada,100)
    t2=time.clock()
    print("El tiempo es: ",t2-t1)
    print("Patrones Orinigales")
    for i in arange(Entrada.shape[0]):
        red.ClasifyPattern(Entrada[i])

    print()
    print()
    Prueba=array([ [3,3,3,3,3],
                   [61,61,61,61,61],
                   [120,120,120,120,120],
                   [275,275,275,275,275],
                   ])
    for i in arange(Prueba.shape[0]):
        red.ClasifyPattern(Prueba[i])



if __name__ == '__main__':
    PruebaSOM()

