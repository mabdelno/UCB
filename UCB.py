#Name: Mary Abdelnour
#Student number: 500648395
import numpy as np
import random
import math
import time

random.seed(time.time())

#returns whether there is a reward or not given which arm is pulled
def reward(percent):
    if random.randrange(100)/100 < percent:
        return 1
    else:
        return 0
    
#Q_a = estimated probabilities q_a = actual probabilities 
#N_a = step count for each action
Q_a, N_a, q_a = np.empty(10), np.empty(10), np.empty(10)

#initializes probabilities of each arm
def initialize(): 
	#step count for all iterations
    n = 1 
    q_a = np.random.random((1,10))[0] 
    #step count for each arm
    N_a = np.zeros(10) 
    #initializes probabilities to be learned and fills array with 50% for each arm
    Q_a = np.empty(10) 
    Q_a.fill(0.5)
    return Q_a, N_a, q_a, n


#implements UCB algorithm by selecting action and updating Q_a, R_a, and N_a
def UCB(n, Q_a, N_a, c = 2, iters=5000):
    R_a = np.zeros(10)
    r_a = np.zeros(10)
    #optimal action count
    oa_count = 0 
    #array of action probabilities
    A_t = np.empty(10) 
    #choose action
    for i in range(n, iters): 
        for a in range(10):
            A_t[a] = Q_a[a] + c*(math.sqrt((math.log(n)) / N_a[a]))
        a_t = np.argmax(A_t)

        action_reward = reward(q_a[a_t])
        #times arm was pulled by agent
        N_a[a_t] = N_a[a_t] + 1
        #keeps record of reward count
        R_a[a_t] = R_a[a_t] + action_reward 
        #averages rewards over amount of times
        r_a[a_t] = R_a[a_t]/N_a[a_t] 
        #update probabilities
        Q_a[a_t] = Q_a[a_t] + (1/N_a[a_t])*(r_a[a_t] - Q_a[a_t])

        #counts how many times optimal action (arm pull) occured
        if np.argmax(q_a) == a_t:
            oa_count += 1
        #print every 100 iterations    
        if n%100 == 0:
            print("Times optimal action chosen:", oa_count)
            rewardpercentages = (sum(R_a)/n)*100
            print("Average reward:", rewardpercentages)
        n += 1
    
    #eoa = estimated optimal action
    eoa = np.argmax(Q_a) 
    #aoa = actual optimal oction
    aoa = np.argmax(q_a)

    return eoa, aoa, rewardpercentages

#total reward percentages
ra = 0
#current reward percentages
rp = 0 
#count of number of times optimal action chosen correctly
oa_chosen = 0
for i in range(100):
    print("Environment #%d\n" % (i+1))
    Q_a, N_a, q_a, n = initialize()
    eoa, aoa, rp = UCB(n, Q_a, N_a)
    if eoa == aoa:
        oa_chosen += 1
    ra = rp + ra
    print("----------------------\n")
ra = ra/100

print("Total reward percentage average with 100 different environments:", ra)
print("Total number of times estimated optimal action = actual optimal action:", oa_chosen)