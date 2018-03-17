#!/usr/bin/env python
import gym
import random
import time
grid_mdp=gym.make('GridWorld-v0')
grid_mdp=grid_mdp.env
grid_mdp.reset()

def mc(num_iter,epsilon):
    n_num=dict()
    qfunc=dict()
    for state in grid_mdp.states:
        for action in grid_mdp.actions:
            qfunc["%d_%s"%(state,action)]=0.0
            n_num["%d_%s"%(state,action)]=0.001
    for iter in range(num_iter):
        state_sample=[]
        action_sample=[]
        reward_sample=[]
        state=grid_mdp.states[int(random.random()*len(grid_mdp.states))]
        done=False
        count=0
        # generate the episode
        while False==done and count<100:
            #action=grid_mdp.actions[int(random.random()*len(grid_mdp.actions))]
            #the evaluate or generating of action is from epsilon_greedy
            action=epsilon_greedy(qfunc,state,epsilon)
            grid_mdp.state=state
            #interact with the environment,get the next_sate ,reward from the environmrnt
            next_s,r,done,info=grid_mdp.step(action)
            state_sample.append(state)
            action_sample.append(action)
            reward_sample.append(r)
            state=next_s
            count+=1
        G=0
        #calculate the G after the episode
        for i in range(len(state_sample)-1,-1,-1):
            G*=grid_mdp.gamma
            G+=reward_sample[i]
        #calculate the Q(s,a) through average the G
        for i in range(len(state_sample)):
            key="%d_%s"%(state_sample[i],action_sample[i])
            n_num[key]+=1.0
            qfunc[key]=qfunc[key]+(G-qfunc[key])/n_num[key]
            G-=reward_sample[i]
            G/=grid_mdp.gamma
    #policy improvement with the greedy policy
    policy_improvement(qfunc)
    #policy_improvement with epsilon_greedy policy
    #policy_improvement_epsilon_greedy(qfunc,epsilon)
    return

def policy_improvement(qfunc):
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
def policy_improvement_epsilon_greedy(qfunc,epsilon):
    for state in grid_mdp.states:
        if state in grid_mdp.terminate_states:continue
        action=epsilon_greedy(qfunc,state,epsilon)
        grid_mdp.pi[state]=action
    return
       
    
def mc_test():
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
    mc(100,0.4)
    #print(grid_mdp.pi)
    mc_test()
    

        
        
        
        
        
        
        
    
           
            
        
            
            
        
    
