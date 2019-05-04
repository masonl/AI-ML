'''
Created on Apr 24, 2019

https://towardsdatascience.com/reinforcement-learning-with-openai-d445c2c687d2
@author: Mason
'''

import gym
from gym import envs
from time import sleep
import numpy as np
from scipy import interpolate

def show_all_envs():
    for i, e in enumerate(envs.registry.all()):
        print(i + 1, '-->', e)

def acrobot():
    # It renders instance of 500 timesteps, perform random actions
    env = gym.make('Acrobot-v1')
    env.reset()
    for _ in range(500):
        env.render()
        env.step(env.action_space.sample())
        sleep(5)
        env.close()
    
def mountain_car_continuous():
    env = gym.make('MountainCarContinuous-v0')
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print('Finished after {} timesteps'.format(t + 1))
            sleep(2)            
    sleep(5)
    env.close()
    
def cart_pole():
    env = gym.make('CartPole-v0')
    env.reset()
    env.render()
    sleep(5)
    env.close()
    
def frozen_lake():
    # 1. Load environment and Q-table structure
    env = gym.make('FrozenLake8x8-v0')
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    # 2. Parameters of Q-learning
    eta = .628
    gma = .9
    epis = 15000
    rev_list = [] # rewards per episode calculate
    
    # 3. Q-learning Algorithm
    for i in range(epis):
        s =env.reset()
        rAll = 0
        d = False
        j = 0
        # The Q-table learning algorithm
        while j < 99:
            # S: starting point, safe)
            # F: frozen surface, safe)
            # H: hole, fall to your doom)
            # G: goal, where the frisbee is located)

            env.render()
            j += 1
            # Choose action from Q table
            a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1./(i + 1)))
            s_n, r, d, _ = env.step(a)
            # Update Q-Table with new knowledge
            Q[s, a] = Q[s, a] + eta * (r + gma * max(Q[s_n, :]) - Q[s, a])
            rAll += r
            s = s_n
            if d == True:
                break
            rev_list.append(rAll)
            env.render()
    
    print('Reward Sum on all episodes ' + str(sum(rev_list)/epis))        
    print('Final Value Q-table')
    max_q = np.max(Q, axis=1)
    env_reward = max_q.reshape(8, 8)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    
    plt.imshow(env_reward, cmap='gray')
    print(env_reward)
    env.close()
def play_frozen_lake():
    env = gym.make('FrozenLake8x8-v0')
    epis = 0
    while True:
        state = env.reset()
        print('state:', state)
        env.render()
        while True:
             # 1 right
             # 2 down
             # 3 up
             # 
             action = env.action_space.sample()
             next_state, reward, done, info = env.step(action)
             print('state:', state, 'action:', action, 'reward:', reward, 'done:', done, 'info:', info)
             state = next_state
             print('next state:', state)
             env.render()
             if done:
                 break
        if reward > 0:
            break
        epis += 1
        if epis % 100 == 0:
            sleep(10)
        
     

if __name__ == '__main__':
    play_frozen_lake()