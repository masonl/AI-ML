"""
Created on Mon Apr 29 16:30:54 2019

@author: Mason

agent
environment
action
state
reward or penalty
policy

1. Observation of the environment
2. Deciding how to act using some strategy
3. Acting accordingly
4. Receiving a reward or penalty
5. Learning from the experiences and refining our strategy
6. Iterate until an optimal stategy is found
"""
import gym
import itertools
import functools
from memory_profiler import profile
from enum import Enum

class Action(Enum):
    '''
    0 = south
    1 = north
    2 = east
    3 = west
    4 = pickup
    5 = dropoff
    '''
    SOUTH = 0
    NORTH = 1
    EAST = 2
    WEST = 3
    PICKUP = 4
    DROPOFF = 5

class PassengerLoc(Enum):
    R = 0
    G = 1
    Y = 2
    B = 3
    TAXI = 4


   
class TaxiEnvDriver():
    # gym.envs.toy_text.taxi.TaxiEnv
    def __init__(self):
        self._env = gym.make('Taxi-v2').env # this 'env' has no 200 steps limit

    @property
    def action_count(self):
        return self._env.action_space.n

    @property
    def state_count(self):
        return self._env.observation_space.n
    
    def render(self, mode='human'):
        return self._env.render(mode)
                
    def reset(self):
        self._env.reset()
        
    def close(self):
        self._env.close()

    @property
    def reward_range(self):
        A = self._env.P.values()
        AV = map(lambda a:a.values(), A)
        AVF = map(lambda av:map(lambda i:i[0], av), AV)
        AVFZ = map(lambda avf:zip(*avf), AVF)
        R = map(lambda a:itertools.islice(a, 2, 3), AVFZ)
        RF = map(lambda r:next(r), R)
        RC = itertools.chain(RF)
        reward_range = set()
        for r in RC:
            reward_range.update(r)
        return reward_range
    
    def set_state(self, taxi_x=3, taxi_y=1, passenger_at=2, destination=0):
        # (taxi row, taxi column, passenger index, destination index)
        state = self._env.encode(taxi_x, taxi_y, passenger_at, destination)     
        self._env.s = state
        return state
