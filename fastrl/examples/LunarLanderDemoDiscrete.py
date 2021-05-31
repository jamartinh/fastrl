from fastrl.algorithms.FARLBasicGYM import FARLBase
import gym
from fastrl.valuefunctions.kNNFaiss import kNNQFaiss
from fastrl.valuefunctions.kNNFaissExt import kNNQFaissExt
import numpy as np
from fastrl.actionselection.ActionSelection import EpsilonGreedyActionSelection
import pickle
import matplotlib.pyplot as plt
import time
import pandas as pd


def Experiment(Episodes=100, nk=1):
    print()
    print('===================================================================')
    print('           INIT EXPERIMENT', 'k=' + str(nk + 1))

    # results of the experiment
    x = list(range(1, Episodes + 1))
    y = []

    # Build the Environment
    # MCEnv = gym.make('MountainCar-v0')
    Env = gym.make('LunarLander-v2')
    n_actions = Env.action_space.n

    # Build a function approximator
    # Q = kNNQ(nactions=MCEnv.nactions,input_ranges=MCEnv.input_ranges,nelemns=[10,5],npoints=False,k=nk+1,alpha=0.5)

    # best
    nk = 27
    # Q = kNNQ(nactions=n_actions, low=np.clip(Env.observation_space.low, -5, 5), high=np.clip(Env.observation_space.high,-5,5), n_elemns=[15, 15, 15, 15], k=nk + 1, alpha=5, lm=0.95)
    n_elems = [8 for _ in range(Env.observation_space.low.shape[0])]
    Q = kNNQFaiss(nactions=n_actions, low=np.clip(Env.observation_space.low, -1, 1), high=np.clip(Env.observation_space.high, -1, 1), n_elemns=n_elems, k=nk, alpha=0.1, lm=0.95)
    #Q = kNNQFaissExt(nactions=n_actions, low=np.clip(Env.observation_space.low, -1, 1), high=np.clip(Env.observation_space.high, -1, 1), n_elemns=n_elems, k=nk, alpha=2, lm=0.95)

    # Get the Action Selector
    As = EpsilonGreedyActionSelection(epsilon=0.1)
    # As = e_softmax_selection(epsilon=0.1)

    # Build the Agent
    Trainer = FARLBase(Q, Env, As, gamma=0.999)
    Trainer.Environment.graphs = True
    # Trainer.Environment.PlotPopulation(Trainer.Q)
    render = False
    fig = plt.figure(figsize=(20, 5))
    fig.canvas.draw()
    ax = fig.add_subplot(1, 1, 1)
    plt.show(block=False)

    for i in range(Episodes):
        t1 = time.perf_counter()
        result = Trainer.sarsa_episode(200, render=render)
        #result = Trainer.q_learning_episode(1000, render=render)
        Q.reset_traces()
        t2 = time.perf_counter() - t1
        As.epsilon *= 0.9
        # Q.alpha *= 0.95

        y.append(result[0])
        miny = min(y)

        if i % 10 == 0:
            ax.plot(range(1, len(y) + 1), y, 'b', label='true')
            ax.plot(range(1, len(y) + 1), pd.Series(y).rolling(20).mean(), 'g', label='soft')
            ax.set_title(f"Episode: {i} reward:{result[0]:.2f}  Steps: {result[1]} alpha: {Q.alpha:.2f} epsilon: {As.epsilon:.2f}")
            ax.grid('on')
            # ax.axis([1, i, 0, 1100])
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Reward')
            fig.canvas.draw()
            # plt.draw()
            plt.pause(0.01)
        print(f"Episode: {i} reward:{result[0]}  Steps: {result[1]} time: {t2} alpha: {Q.alpha} epsilon: {As.epsilon}")
        if i >= 1900:
            render = True

    return x, y, nk


if __name__ == '__main__':
    Experiment(Episodes=200000, nk=15)
    # Experiments()
