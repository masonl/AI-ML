'''
Created on Apr 24, 2019

@author: Mason
'''
import gym

def ms_pacman():
    env = gym.make("MsPacman-v0")
    state = env.reset()
    print('Number of actions:', env.action_space.n)
    print('Number of states:', env.observation_space)
    print('Action meaning:')
    print(env.env.get_action_meanings())
    env.render()
    env.close()
    
def random_sample():
    env = gym.make("MsPacman-v0")
    state = env.reset()
    reward, info, done = None, None, None
    while done != True:
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()
    env.close()

if __name__ == '__main__':
    ms_pacman()