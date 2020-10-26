import gym

from fastrl.valuefunctions.FAInterface import FARL as fai


class FARLBase:

    def __init__(self, function_approximator: fai, env: gym.Env, action_selector, gamma=1.0):

        self.gamma = gamma  # discount factor
        self.Environment = env
        self.n_actions = env.action_space.n  # number of actions
        self.Q = function_approximator  # the function approximator
        self.SelectAction = action_selector  # the action_selection function
        self.SelectAction.parent = self

    def sarsa_episode(self, maxsteps=100, render=False):
        # do one episode with sarsa learning
        # maxstepts: the maximum number of steps per episode
        # Q: the current QTable
        # alpha: the current learning rate
        # gamma: the current discount factor
        # epsilon: probablity of a random action
        # statelist: the list of states
        # actionlist: the list of actions

        s = self.Environment.reset()
        steps = 0
        total_reward = 0
        r = 0
        # selects an action using the epsilon greedy selection strategy
        a, v = self.SelectAction(s)

        for i in range(1, maxsteps + 1):

            # do the selected action and get the next car state
            sp, r, done, info = self.Environment.step(a)

            # observe the reward at state xp and the final state flag
            total_reward = total_reward + r

            # select action prime
            ap, vp = self.SelectAction(sp)

            # update the Qtable, that is,  learn from the experience
            target_value = r + self.gamma * vp * (not done)
            # self.Q.add_trace(s,a,r)
            self.Q.update(s, a, target_value)

            # update the current variables
            s = sp
            a = ap

            # increment the step counter.
            steps = steps + 1
            if render:
                self.Environment.render()

            # if reachs the goal breaks the episode
            if done:
                break

        return total_reward, steps

    def kNNCQEpisode(self, maxsteps=100):
        # do one episode with sarsa learning
        # maxstepts: the maximum number of steps per episode
        # Q: the current QTable
        # alpha: the current learning rate
        # gamma: the current discount factor
        # epsilon: probablity of a random action
        # statelist: the list of states
        # actionlist: the list of actions

        s = self.Environment.reset()
        steps = 0
        total_reward = 0
        r = 0
        # selects an action using the epsilon greedy selection strategy

        action, a = self.Environment.step(s)

        for i in range(1, maxsteps + 1):

            # do the selected action and get the next car state
            sp = self.Environment.DoAction(action, s)

            # observe the reward at state xp and the final state flag
            r, isfinal = self.Environment.GetReward(sp, action)
            total_reward = total_reward + r

            # select action prime
            actionp, ap = self.Q.GetActionList(sp)

            # update the Qtable, that is,  learn from the experience
            target_value = r + self.gamma * self.Q(sp) * (not isfinal)
            self.Q.update(s, a, target_value)

            # update the current variables
            s = sp
            a = ap
            action = actionp

            # increment the step counter.
            steps = steps + 1

            # if reachs the goal breaks the episode
            if isfinal == True:
                break

        return total_reward, steps

    def q_learning_episode(self, maxsteps=100, render=True):
        """ do one episode of QLearning """
        # maxstepts: the maximum number of steps per episode
        # alpha: the current learning rate
        # gamma: the current discount factor
        # epsilon: probablity of a random action
        # actionlist: the list of actions

        s = self.Environment.reset()
        steps = 0
        total_reward = 0

        for i in range(1, maxsteps + 1):

            # selects an action using the epsilon greedy selection strategy
            a = self.SelectAction(s)

            # do the selected action and get the next state
            sp, r, done, info = self.Environment.step(a)

            # observe the reward at state xp and the final state flag
            total_reward = total_reward + r

            # update the Qtable, that is,  learn from the experience
            vp = self.Q(sp)
            target_value = r + self.gamma * max(vp) * (not done)
            sp = self.Q.update(s, a, target_value)

            # update the current variables
            s = sp

            # increment the step counter.
            steps = steps + 1
            if render:
                self.Environment.render()

            # if reachs the goal breaks the episode
            if done:
                break

        return total_reward, steps
