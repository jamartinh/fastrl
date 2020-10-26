# from kNNQ import kNNQ
# from algorithms.kNNQC import kNNQC
from fastrl.valuefunctions.ExaSCIPY import Exa as kNNQC
# from NeuroQ import NeuroQ
# from RNeuroQ import RNeuroQ
# from SNeuroQ import SNeuroQ
# from SOMQ import SOMQ


from fastrl.actionselection.ActionSelection import *
import pickle
# from pylab import *
import time


def CartPoleExperiment(Episodes=100, nk=0):
    print()
    print('===================================================================')
    print('           INIT EXPERIMENT', 'k=' + str(nk + 1))

    # results of the experiment
    x = list(range(1, Episodes + 1))
    y = []
    yr = []

    # Build the Environment
    Env = CartPoleEnvironment()

    # Build a function approximator
    # Q = kNNQ(nactions=Env.nactions,input_ranges=Env.input_ranges,nelemns=[2,3,10,2],npoints=False,k=1,alpha=0.25)
    # Q = kNNQ(nactions=Env.nactions,input_ranges=Env.input_ranges,nelemns=[2+7,3+7,10+3,2+7],npoints=False,k=nk+1,alpha=0.3,lm=0.95)
    # Q =  NeuroQ(Env.nactions, Env.input_ranges, 20, Env.reward_ranges,Env.deep_in,Env.deep_out,alpha=0.3)
    # Q = SNeuroQ(Env.nactions, Env.input_ranges, 6, Env.output_ranges,alpha=0.2)

    # Experiments
    # Q = kNNQC(Env.input_ranges,[2+2,3+2,10+1,2+1],Env.output_ranges,[11],nk+1,0.3,0.90) #excelent

    # BEST
    Q = kNNQC(Env.input_ranges, [2 + 2, 3 + 2, 10 + 1, 2 + 1], Env.output_ranges, [11], 4, 0.6, 0.90, 0.0)  # excelent
    # Q = kNNQC(Env.input_ranges,[2+4,3+4,10+3,2+4],Env.output_ranges,[11],9,0.3,0.90) #notbad
    # Q = kNNQC(Env.input_ranges,[10,10,10,10],Env.output_ranges,[11],32,2.0,0.90) #good

    # Get the Action Selector
    As = EpsilonGreedyActionSelection(epsilon=0.0)
    # As = e_softmax_selection(epsilon=0.1)
    # As = None
    # Build the Agent
    CP = FARLBase(Q, Env, As, gamma=1.0)
    n_actions = Env.action_space.sample.shape[0]

    for i in range(Episodes):
        # result = CP.sarsa_episode(1000)
        # result  = CP.NeuroQEpisode(1000)
        t1 = time.clock()
        result = CP.kNNCQEpisode(1000)
        t2 = time.clock()
        # result = CP.q_learning_episode(1000)
        CP.SelectAction.tau = CP.SelectAction.tau * 0.9
        CP.PlotLearningCurve(i, result[1], CP.SelectAction.tau)
        print("Episode:", str(i), 'Total Reward:', str(result[0]), 'Steps:', str(result[1]), "time", t2 - t1)

        y.append(result[1])
        yr.append(result[0])
    ##        if i==50:
    ##            miny =min(y)
    ##            figure(i)
    ##            plot(range(1,len(y)+1),y,'k')
    ##            title(r'$ k = 4, \quad  \lambda=0.9, \quad  \alpha=0.3 $')
    ##            grid('on')
    ##            axis([1, i, 0, 1100])
    ##            xlabel('Episodes')
    ##            ylabel('Steps')
    ##            savefig('cpresultcontinuous.pdf')
    ##            print "salvado"
    ##            close(i)

    CP.LearningCurveGraph.display.visible = False

    return [[x, y, nk], [x, yr, nk]]


def Experiments():
    results1 = []
    results2 = []
    for i in range(0, 10):
        x = CartPoleExperiment(Episodes=200, nk=i)
        results1.append(x[0])
        results2.append(x[1])

    pickle.dump(results1, open('cartpolestepscq.dat', 'w'))
    pickle.dump(results2, open('cartpolerewardcq.dat', 'w'))


if __name__ == '__main__':
    # Experiments()
    x = CartPoleExperiment(50, 3)
    pickle.dump(x[0], open('deprecated/contiuouscartpolesteps.dat', 'w'))
