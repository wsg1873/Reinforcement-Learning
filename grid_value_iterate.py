#!/usr/bin/env python
import gym
import time
import random


grid_mdp=gym.make('GridWorld-v0')
grid_mdp=grid_mdp.env
grid_mdp.reset()

def value_iterate():
    # iterate 1000 times
    for i in range(1000):
        #traverse the state
        for state in grid_mdp.states:
            grid_mdp.state=state
            a=grid_mdp.actions[0]
            #interact with the environment,get the next_sate ,reward from the environmrnt
            next_s,r,done,info=grid_mdp.step(a)
            v=r+grid_mdp.gamma*grid_mdp.v[next_s]
            #evaluate,calculate the maximum value function
            for action in grid_mdp.actions:
                grid_mdp.state=state
                next_s,r,done,info=grid_mdp.step(action)                
                if v<r+grid_mdp.gamma*grid_mdp.v[next_s]:
                    a=action
                    v=r+grid_mdp.gamma*grid_mdp.v[next_s]
            #record the best action
            grid_mdp.pi[state]=a
            #record the new value function
            grid_mdp.new_v[state]=v
        grid_mdp.v=grid_mdp.new_v
    #test the value_iterate
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
    value_iterate()
                
            
        
            
