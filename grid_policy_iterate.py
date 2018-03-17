#!/usr/bin/env python
#fuction：policy_itreate 
#@author: Shiguang.Wu
import gym
import random
import time
from gym import wrappers
grid_mdp=gym.make('GridWorld-v0') 
grid_mdp=grid_mdp.env

def policy_iterate():
    policy_evaluate();
    policy_improve();
#policy evaluate 100 times get the value function
def policy_evaluate():
    for i in range(100):
        for state in grid_mdp.states:
            if state in grid_mdp.terminate_states: continue
            new_sum=0
            for action in grid_mdp.actions:
                grid_mdp.state=state
                next_s,r,done,info=grid_mdp.step(action)
                temp_v=r+grid_mdp.gamma*grid_mdp.v[next_s]
                new_sum+=temp_v
            grid_mdp.new_v[state]=new_sum/4
        grid_mdp.v=grid_mdp.new_v
    return
#policy improvement get the best action in every state
def policy_improve():
    for state in grid_mdp.states:
        if state in grid_mdp.terminate_states: continue
        a=grid_mdp.actions[0]
        next_s,r,done,info=grid_mdp.step(a)
        v1=r+grid_mdp.gamma*grid_mdp.v[next_s]
        for action in grid_mdp.actions:
            grid_mdp.state=state
            next_s,r,done,info=grid_mdp.step(action)
            if v1<r+grid_mdp.gamma*grid_mdp.v[next_s]:
                a=action
                v1=r+grid_mdp.gamma*grid_mdp.v[next_s]
        grid_mdp.pi[state]=a
    return grid_mdp.pi
#test 
def test():
    grid_mdp.state=grid_mdp.states[random.randint(1,5)]
    grid_mdp.render()
    while 1:
        time.sleep(1)#delay 1 second
        action=grid_mdp.pi[grid_mdp.state]
        next_s,r,done,info=grid_mdp.step(action)
        grid_mdp.render()
        if done:
            break
    return 
            
if __name__=="__main__":
    policy_iterate()
    test()
        
        
        
    
                 
             
