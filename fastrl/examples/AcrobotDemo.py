import time

import pickle

from Environments.AcrobotEnvironmentG import AcrobotEnvironment
from fastrl.actionselection.ActionSelection import *
from deprecated.FARLBasic import *
from fastrl.valuefunctions.kNNSCIPY import kNNQ


def AcrobotExperiment(Episodes = 100, nk = 1):
    print()
    print('===================================================================')
    print('           INIT EXPERIMENT', 'k=' + str(nk))

    # results of the experiment
    x = list(range(1, Episodes + 1))
    y = []

    # Build the Environment
    ACEnv = AcrobotEnvironment()

    # Build a function approximator
    Q = kNNQ(nactions = ACEnv.nactions, input_ranges = ACEnv.input_ranges, nelemns = [11, 11, 11, 11], npoints = False, k = 2 ** 4, alpha = 5.0, lm = 0.90)

    # Q.Q+=10000
    # Q = kNNQ(nactions=ACEnv.nactions,input_ranges=ACEnv.input_ranges,nelemns=False,npoints=300,k=5,alpha=0.3)
    # Q = NeuroQ(ACEnv.nactions, ACEnv.input_ranges, 30+nk, ACEnv.reward_ranges,ACEnv.deep_in,ACEnv.deep_out,alpha=0.3)
    # Q = RNeuroQ(MCEnv.nactions, MCEnv.input_ranges, 10, MCEnv.reward_ranges,alpha=0.3)
    # Q = SOMQ(nactions=MCEnv.nactions,size_x=20,size_y=20,input_ranges=MCEnv.input_ranges,alpha=0.3)
    # Q = lwprQ(nactions=ACEnv.nactions,input_ranges=ACEnv.input_ranges)
    # Get the Action Selector
    As = EpsilonGreedyActionSelection(epsilon = 0.000)
    # As = e_softmax_selection(epsilon=0.1)

    # Build the Agent
    AC = FARLBase(Q, ACEnv, As, gamma = 1.0)

    AC.Environment.graphs = True  # False
    # AC.Environment.PlotPopulation(MC.Q)

    for i in range(Episodes):
        t1 = time.clock()
        result = AC.SARSAEpisode(1000)
        # result = AC.q_learning_episode(1000)
        t2 = time.clock() - t1
        # AC.SelectAction.epsilon = AC.SelectAction.epsilon * 0.9
        AC.PlotLearningCurve(i, result[1], AC.SelectAction.tau)
        # AC.Environment.PlotPopulation(MC.Q)
        print('Episode', str(i), ' Steps:', str(result[1]), 'time', t2)
        y.append(result[1])

    return [x, y, nk]


def Experiments():
    results = []
    for i in range(0, 10):
        x = AcrobotExperiment(Episodes = 1000, nk = i)
        results.append(x)

    pickle.dump(results, open('acrobotresult1.dat', 'w'))


if __name__ == '__main__':
    AcrobotExperiment(Episodes = 1001, nk = 0)
    # Experiments()
