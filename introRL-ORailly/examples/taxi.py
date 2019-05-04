'''
Created on Apr 22, 2019

@author: Mason
'''

import gym
import numpy as np
from scipy.constants import golden
from gym.envs.tests.test_envs_semantics import episodes

def taxi():
    #
    # the yellow square represents the taxi, 
    # the (“|”) represents a wall, 
    # the blue letter represents the pick-up location, 
    # the purple letter is the drop-off location. 
    # The taxi will turn green when it has a passenger aboard. 
    #
    env = gym.make('Taxi-v2')
    print('Number of states:', env.observation_space.n)
    #
    # down (0), up (1), right (2), left (3), pick-up (4), and drop-off (5)
    #
    print('Number of actions:', env.action_space.n)
    
    env.reset()
    env.env.s = 114
    print('render when state = {}'.format(env.env.s))
    env.render()
    state, reward, done, info = env.step(1)
    print('return after one step up:', state, reward, done, info)
    env.render()
    
    print('solve the env using random actions:')
    state = env.reset()
    counter = 0
    reward = None
    while reward != 20:
        state, reward, done, info = env.step(env.action_space.sample())
        counter += 1
    
    print('Total steps to solve:', counter)
    
    print('Calculate Q action value table')
    # Q action value table.
    Q = np.zeros([env.observation_space.n, env.observation_space.n, env.action_space.n])
    # total accumulated reward for each episode
    G = 0
    # learning rate
    alpha = golden / 2
    wins = 0
    total_G = 0
    
    for  episode in range(1, 25001):
        done = False
        G, reward = 0, 0
        state = env.reset()
        init_state = state
        counter = 0
        while done != True:
            # choosing an action with the highest Q value 
            # for the current state 
            action = np.argmax(Q[init_state, state])
            next_state, reward, done, info = env.step(action)
            counter += 1
            #
            # using the action value formula (based upon the Bellman equation) 
            # and allows state-action pairs to be updated in a recursive fashion 
            # (based on future values)
            #
            if episode <= 15_000: 
                Q[init_state, state, action] += alpha * (
                                         reward + 
                                         np.max(Q[init_state, next_state]) - 
                                         Q[init_state, state, action]
                                        )
            G += reward
            state = next_state
            
        total_G += G
        #if episode % 50 == 0:
        if reward == 20:
            wins += 1
         
        if episode % 100 == 0:
            print(episode, str(wins) + '% success rate.', 'everage G:', total_G / 100.)
            wins = 0
            total_G = 0
            #print('Episode {} Total Reward: {}, current reward: {}, steps: {}, info: {}, done: {}'.format(episode, G, reward, counter, info, done))

    #print('Q action table:')
    #print(Q)
    wins = 0
    total_G = 0
    G = 0
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    for episode in range(1,25001):
        done = False
        G, reward = 0,0
        state = env.reset()
        while done != True:
                action = np.argmax(Q[state]) #1
                state2, reward, done, info = env.step(action) #2
                if episode <= 15_000:
                    Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action]) #3
                G += reward
                state = state2   
#         if episode % 50 == 0:
#             print('Episode {} Total Reward: {}'.format(episode,G))
        total_G += G
        if reward == 20:
            wins += 1
        if episode % 100 == 0:
            print(episode, str(wins) + '% success rate.', 'everage G:', total_G / 100.)
            wins = 0
            total_G = 0
    
    
if __name__ == '__main__':
    taxi()
