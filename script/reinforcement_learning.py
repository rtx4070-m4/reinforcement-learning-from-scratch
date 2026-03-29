
import gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Q-Learning for FrozenLake
# -----------------------------

env = gym.make("FrozenLake-v1", is_slippery=False)

state_size = env.observation_space.n
action_size = env.action_space.n

q_table = np.zeros((state_size, action_size))

alpha = 0.8
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

episodes = 2000
rewards = []

for episode in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    while not done:

        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)

        q_table[state,action] = q_table[state,action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state,action]
        )

        state = next_state
        total_reward += reward

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    rewards.append(total_reward)

print("Q-Learning training completed.")

# -----------------------------
# Deep Q-Network for CartPole
# -----------------------------

env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

memory = deque(maxlen=2000)

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32

def build_model():
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate))
    return model

model = build_model()

episodes = 300

for e in range(episodes):

    state = env.reset()[0]
    state = np.reshape(state,[1,state_size])

    for time in range(500):

        if np.random.rand() <= epsilon:
            action = random.randrange(action_size)
        else:
            action = np.argmax(model.predict(state, verbose=0)[0])

        next_state, reward, done, _, _ = env.step(action)
        next_state = np.reshape(next_state,[1,state_size])

        memory.append((state,action,reward,next_state,done))
        state = next_state

        if done:
            print("episode:",e,"score:",time)
            break

        if len(memory) > batch_size:

            minibatch = random.sample(memory,batch_size)

            for s,a,r,ns,d in minibatch:

                target = r

                if not d:
                    target = r + gamma * np.amax(model.predict(ns,verbose=0)[0])

                target_f = model.predict(s,verbose=0)
                target_f[0][a] = target

                model.fit(s,target_f,epochs=1,verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("DQN training completed.")
