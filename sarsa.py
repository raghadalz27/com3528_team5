## CLASS IMPLEMENTING SARSA RL ALGORITHM


import numpy as np
import matplotlib.pyplot as plt

class SARSA:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        # epsilon-greedy policy for action selection (exploration vs exploitation)
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q[state, :])
        return action
    
    def choose_best_action(self, state):
        # greedy policy for action selection (exploitation)
        action = np.argmax(self.Q[state, :])
        return action


    def update(self, state, action, reward, next_state, next_action):
        # update Q-table using SARSA algorithm (improve its policy)
        td_error = reward + self.gamma * self.Q[next_state, next_action] - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def learn(self, env, n_episodes=1000, max_steps=1000):
        # learn Q-table from environment
        # env is an instance of the environment
        # env.step(action) needs to return next_state, reward, done, info
        # env.reset() needs to return env to initial state
        rewards_per_episode = []
        steps_per_episode = []
        for episode in range(n_episodes):
            state = env.reset()[0]
            action = self.choose_action(state)
            rewards = 0
            steps = 0
            for step in range(max_steps):
                print(env.step(action))
                next_state, reward, done, _, _ = env.step(action)
                next_action = self.choose_action(next_state)
                self.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
                rewards += reward
                steps += 1
                if done:
                    break
            rewards_per_episode.append(rewards)
            steps_per_episode.append(steps)

        # Plot rewards and steps per episode
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards_per_episode)
        plt.title('Rewards per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.subplot(1, 2, 2)
        plt.plot(steps_per_episode)
        plt.title('Steps per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.show()
