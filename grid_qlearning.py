#!/usr/bin/env python
import gym
import random
import time

grid_mdp=gym.make('GridWorld-v0')
grid_mdp=grid_mdp.env
grid_mdp.reset()

def qlearning(num_iter,alpha,epsilon):
    qfunc=dict()# action value function
    #initial the action value function
    for state in grid_mdp.states:
        for action in grid_mdp.actions:
            key="%d_%s"%(state,action)
            qfunc[key]=0.0
    #iter
    for iter in range(num_iter):
        #initial the state,action,
        state=grid_mdp.reset()
        action=epsilon_greedy(qfunc,state,epsilon)
        done=False
        count=0
        #until the one iter end
        while False==done and count<100:
            key="%d_%s"%(state,action)
            grid_mdp.state=state
            #interact with the environment,get the next_sate ,reward from the environmrnt
            next_s,r,done,info=grid_mdp.step(action)
            #get the action in next_s
            next_a=greedy(qfunc,next_s)#evaluate policy is greedy
            key_next="%d_%s"%(next_s,next_a)
            qfunc[key]=qfunc[key]+alpha*(r+grid_mdp.gamma*qfunc[key_next]-qfunc[key])
            state=next_s
            action=epsilon_greedy(qfunc,state,epsilon)#action policy is epsilon greedy
            count+=1
    #policy improvemnt
    policy_improvemnt(qfunc)
    return qfunc
# q learning applying in robot looking a gold 
def qlearning_test():
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
#policy improvement get the best policy 
#PI(s) is that get a s.t maxminze Q(s,a)
def policy_improvemnt(qfunc):
    for state in grid_mdp.states:
        if state in grid_mdp.terminate_states:continue
        key="%d_%s"%(state,grid_mdp.actions[0])
        Q=qfunc[key]
        a=grid_mdp.actions[0]
        for action in grid_mdp.actions:
            key="%d_%s"%(state,action)
            if Q<qfunc[key]:
                a=action
                Q=qfunc[key]
        grid_mdp.pi[state]=a
    return            
# greedy policy
def greedy(qfunc,state):
    action_que=0
    key="%d_%s"%(state,grid_mdp.actions[0])
    qmax=qfunc[key]
    for i in range(len(grid_mdp.actions)):
        key="%d_%s"%(state,grid_mdp.actions[i])
        if qmax<qfunc[key]:
            qmax=qfunc[key]
            action_que=i
    return grid_mdp.actions[action_que]

#epsilon_greedy policy
def epsilon_greedy(qfunc, state, epsilon):
    key="%d_%s"%(state,grid_mdp.actions[0])
    pro=[0.0 for i in range(len(grid_mdp.actions))]
    qmax=qfunc[key]
    action_que=0
    for i in range(len(grid_mdp.actions)):
        key="%d_%s"%(state,grid_mdp.actions[i])
        pro[i]=epsilon/len(grid_mdp.actions)
        if qmax<qfunc[key]:
            qmax=qfunc[key]
            action_que=i
    pro[action_que]+=1-epsilon
    r= random.random()
    s=0.0
    for i in range(len(grid_mdp.actions)):
        s+=pro[i]
        if s>=r:return grid_mdp.actions[i]
    return grid_mdp.actions[len(grid_mdp.actions)-1]

if __name__=="__main__":
    qlearning(100,0.8,0.4)
    qlearning_test()
    
    
    
    
    
    
    
    
    
    
