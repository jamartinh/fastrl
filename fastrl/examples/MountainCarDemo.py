from algorithms.FARLBasicGYM import FARLBase
import gym
from fastrl.valuefunctions.kNNSCIPY import kNNQ
from fastrl.actionselection.ActionSelection import EpsilonGreedyActionSelection
from fastrl.valuefunctions.kNNFaiss import kNNQFaiss
import time


def MountainCarExperiment(Episodes=100, nk=1):
    print()
    print('===================================================================')
    print('           INIT EXPERIMENT', 'k=' + str(nk + 1))

    # results of the experiment
    x = list(range(1, Episodes + 1))
    y = []

    # Build the Environment
    Env = gym.make('MountainCar-v0')

    n_actions = Env.action_space.n

    # Build a function approximator
    # Q = kNNQ(nactions=Env.nactions,input_ranges=Env.input_ranges,nelemns=[10,5],npoints=False,k=nk+1,alpha=0.5)

    # best
    nk = 10
    #Q = kNNQ(nactions=n_actions, low=Env.observation_space.low, high=Env.observation_space.high, n_elemns=[15, 15], k=9, alpha=1.0, lm=0.95)
    Q = kNNQFaiss(nactions=n_actions, low=Env.observation_space.low, high=Env.observation_space.high, n_elemns=[15, 15], k=9, alpha=1.0, lm=0.95)

    # Get the Action Selector
    As = EpsilonGreedyActionSelection(epsilon=1.0)
    # As = e_softmax_selection(epsilon=0.1)

    # Build the Agent
    MC = FARLBase(Q, Env, As, gamma=0.999)
    MC.Environment.graphs = True
    # MC.Environment.PlotPopulation(MC.Q)
    render = False
    for i in range(Episodes):
        t1 = time.perf_counter()
        result = MC.sarsa_episode(1000, render=render)
        Q.reset_traces()
        # result = MC.q_learning_episode(1000)
        t2 = time.perf_counter() - t1
        As.epsilon *= 0.9
        Q.alpha *= 0.995

        # MC.PlotLearningCurve(i, result[1], MC.SelectAction.epsilon)
        # MC.Environment.PlotPopulation(MC.Q)
        print('Episode', i, ' Steps:', result[1], 'time:', t2, 'alpha:', Q.alpha, 'epsilon:', As.epsilon)
        if i >= 1900:
            render = True
        y.append(result[1])
    ##        if i==50:
    ##
    ##            figure(i)
    ##            plot(range(1,len(y)+1),y,'k')
    ##            title(r'$ \min=105,\quad k = 4, \quad  \lambda=0.95, \quad  \epsilon=0.0, \quad \alpha=0.9 $')
    ##            grid('on')
    ##            axis([1, i, 0, 800])
    ##            xlabel('Episodes')
    ##            ylabel('Steps')
    ##            savefig('mcresult.pdf')
    ##            print "salvado"
    ##            close(i)


if __name__ == '__main__':
    MountainCarExperiment(Episodes=2000, nk=15)
