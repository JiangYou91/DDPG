# -*- coding: utf-8 -*-
"""

@author: sigaud
"""
import random
from collections import deque
from minibatch import minibatch
from sample import sample
import numpy as np 
import heapq
import tensorflow as tf

class replay_buffer(object):
    """
    A sample from an agent environment interaction
    """
    def __init__(self, min_filled,size):
        self.buffer = deque([])
        self.bests = []
        self.size = size
        self.min_filled = min_filled
        self.reward_min = float("Inf")
        self.reward_max = -float("Inf")
#        
#        self.sorted_buffer= deque([]) 
#        self.td_error= deque([])
        
        self.sorted_buffer= [] #type heap
        self.alpha = 0.3
        
        self.distribution=[]
        
        self.k=64
        self.dist_sur_k=[] 
        self.range_dist=0
        self.length_range=0        
        
    def flush(self):
#        self.buffer = []
        self.sorted_buffer = []
            
    def current_size(self):
#        return len(self.buffer)
        return len(self.sorted_buffer) 
        
    def store_samples_from_batch(self, s_t, a_t, r_t, s_t_next):
        self.reward_max = max(r_t, self.reward_max)
        self.reward_min = min(r_t, self.reward_min)   
        print self.reward_min, self.reward_max
        for i in range(len(s_t)):
            if (self.isFull()):
                self.buffer[random.randint(self.size/5, self.size-1)] = [list(s_t[i]), list(a_t[i]), r_t[i], list(s_t_next[i])]
            else:
                self.buffer.append([list(s_t[i]), list(a_t[i]), r_t[i], list(s_t_next[i])])
        
    def store_one_sample(self, sample):
        self.reward_max = max(sample.reward, self.reward_max)
        self.reward_min = min(sample.reward, self.reward_min)   
        if len(self.bests)==0 or self.bests[-1].reward < sample.reward:
            i=1
            while i<=len(self.bests) and self.bests[-i].reward < sample.reward:
                i+=1
            self.bests.insert(len(self.bests)-i, sample)
            if len(self.bests)>1000:
                self.bests.pop()
#        if (self.isFull()):
#            # replace an older sample, but protecting the beginning
#            i =random.randint(self.size/5,len(self.buffer)-1)
#            self.buffer[i] = sample
#            
##            self.sorted_buffer[i] = sample
##            self.td_error[i] = 1.0
#        else:
#            self.buffer.append(sample)
#             
# 
#        if len(self.td_error)<3:
#            self.td_error.append(1)
#            self.sorted_buffer.append(sample)  
#        else:
##        si tous td error est convergente, le mid td error doit etre aussi convergente
#            l = len(self.td_error) /10/ self.k 
#            mid_td_err = self.td_error[l]
#            self.td_error.insert(l,  mid_td_err)
#            self.sorted_buffer.insert(l, sample)  
                
#        if (self.isFull()): 
#            i =random.randint(0,self.size/5)
#            self.sorted_buffer[i] = sample 
#            self.td_error[i] = self.td_error[len(self.td_error)/10/self.k]
#        elif len(self.td_error)<1:
#            self.td_error.append(0.7)
#            self.sorted_buffer.append(sample)
#        else:
#            self.td_error.append(self.td_error[0])
#            self.sorted_buffer.append(sample)
            
        if (self.isFull()):  
#            heapq.heapreplace(self.sorted_buffer, (np.random.uniform(0,self.sorted_buffer[0][0]),sample))  
             self.sorted_buffer[random.randint(self.size/5,len(self.sorted_buffer)-1)] = (np.random.uniform(0,self.sorted_buffer[0][0]),sample)   
        elif len(self.sorted_buffer)<1:
            heapq.heappush(self.sorted_buffer, (0.7,sample)) 
        else:
            heapq.heappush(self.sorted_buffer, (np.random.uniform(0,self.sorted_buffer[0][0]),sample)) 
            
#        if len(self.sorted_buffer) > 3000:
#        if (self.isFull()):
#            self.sorted_buffer.pop()
#            self.td_error.pop()
#        else:
        if not (self.isFull()):
            self.distribution.append(pow(1.0/len(self.sorted_buffer),self.alpha))  
            self.k=max(64,int(np.sqrt(len(self.distribution))))
            self.dist_sur_k=[]
            for i in range(self.k):
                self.dist_sur_k.append(sum(self.distribution[i*self.k:(i+1)*self.k]))
            
            s=sum(self.dist_sur_k)
            self.dist_sur_k = [i/s for i in self.dist_sur_k]
            self.range_dist = range(len(self.dist_sur_k)) 
            self.length_range= min(1,int(len(self.distribution)/self.k))
            

    def isFullEnough(self):
        return (self.current_size()>self.min_filled)

    def isFull(self):
        return (self.current_size()>=self.size)

    def get_state(self,index):
#        return self.buffer[index].state
#        return self.sorted_buffer[index].state
        return self.sorted_buffer[index][1].state

    def get_random_minibatch(self,batch_size):
            states = []
            rewards = []
            actions = []
            next_states = []
            for i in range(batch_size):
                if random.uniform(0.0,1.0)<0.2:
                    index= random.randint(0, len(self.bests)-1)
                    sample = self.bests[index]
                else:
                    index= random.randint(0, self.current_size()-1)
                    sample = self.buffer[index]
                states.append(sample.state) #no need to put into [] because it is already a vector
                actions.append(sample.action) #no need to put into [] because it is already a vector
                
                if self.reward_max-self.reward_min == 0:
                    rewards.append([sample.reward])
                else:                
                    rewards.append([(sample.reward-self.reward_min)/(self.reward_max-self.reward_min)*2.0-1.0])
                #print((sample.reward-self.reward_min)/(self.reward_max-self.reward_min)*2.0-1.0)
                next_states.append(sample.next_state) #no need to put into [] because it is already a vector
            return minibatch(states,actions,rewards,next_states)
        
#    def init_td_error(self):
        
        #mis a jour des distributions selon la taille de la liste
        
    def get_td_error_sorted_minibatch(self,batch_size):
        
            states = []
            rewards = []
            actions = []
            next_states = []
            self.sample_minibatch=[]
#            sortd_buff = sorted(self.sorted_buffer.items(),key=lambda x: x[1],reverse=True) 
#            sortd_buff_by_reward = sorted(self.sorted_buffer.items(),key=lambda x: x[0].reward,reverse=True) 
             
#            if self.alpha==0:
#                self.alpha=2
            #normalise the distribution 
#            
#            s=sum(self.distribution)
#            dist = [i/s for i in self.distribution]
#            range_buffer = range(len(self.sorted_buffer))            
             
             
            for i in range(batch_size): 
                if np.random.rand()<0.3:
#                    index= min(len(sortd_buff_by_reward)-1,int(np.random.exponential(1)*len(sortd_buff_by_reward)/6))
#                    sample = sortd_buff_by_reward[index][0]
#                    index= random.randint(0, len(self.bests)-1)
#                    index= min(len(self.bests)-1,int(np.random.exponential(1)*len(self.bests)/6))
#                    sample = self.bests[index]
                    index= random.randint(0, len(self.bests)-1)
                    sample = self.bests[index]
                    self.sample_minibatch.append((0,index))
#                    
                else:
#                    index= min(len(sortd_buff)-1,int(np.random.exponential(1)*len(self.sorted_buffer)/6))
#                    index= np.random.choice(range_buffer, p=dist) 
#                    sample = self.sorted_buffer[index]

#                    index_range = np.random.choice(range_dist, p=dist)          
                    index_range = np.random.choice(self.range_dist, p=self.dist_sur_k)  
                    index = np.random.randint(index_range*self.length_range, (index_range+1)*self.length_range)
    #                    sample = self.sorted_buffer[index]               
                    sample = self.sorted_buffer[index][1]              
                    print index, index_range
                    
                    self.sample_minibatch.append((1,index))
                states.append(sample.state) #no need to put into [] because it is already a vector
                actions.append(sample.action) #no need to put into [] because it is already a vector

                if self.reward_max-self.reward_min == 0:
                    rewards.append([sample.reward])
                else:
                    rewards.append([(sample.reward-self.reward_min)/(self.reward_max-self.reward_min)*2.0-1.0])
                #print((sample.reward-self.reward_min)/(self.reward_max-self.reward_min)*2.0-1.0)
                next_states.append(sample.next_state) #no need to put into [] because it is already a vector
            
           
            return minibatch(states,actions,rewards,next_states)
    def update_td_error(self,td_err):
        for i in range(len(td_err)): 
#            self.td_error.pop(self.sample_minibatch[i])
#            sample = self.sorted_buffer.pop(self.sample_minibatch[i])
            
#            j=1
##            find the position by td error of the sample 
#            while j<=len(self.sorted_buffer) and self.td_error[-j] < td_err[i]:  
#                 j+=1
#            a=0;b=len(self.td_error)-1
#            j=int((a+b)/2) 
#            done =False
#            while not done and a<b-1:
#                if self.td_error[j+1] > td_err[i]:
#                    a=j+1
#                    j=int((b+j+1)/2)
#                elif self.td_error[j-1] < td_err[i]:
#                    b=j-1
#                    j=int((a+j-1)/2)
#                else:
#                    done =True;
#                        
#                        
#            self.td_error.insert(j, td_err[i])
#            self.sorted_buffer.insert(j, sample)
#            self.td_error[self.sample_minibatch[i]]=td_err[i]
#            self.sorted_buffer.append(sample)
#            self.sorted_buffer[self.sample_minibatch[i]]=(td_err[i][0],self.sorted_buffer[self.sample_minibatch[i]][1])
            if self.sample_minibatch[i][0]==0:#from bests
                heapq.heappushpop(self.sorted_buffer, (td_err[i][0],self.bests[self.sample_minibatch[i][1]]))
            else:# from buffer
                self.sorted_buffer[self.sample_minibatch[i][1]]=(td_err[i][0],self.sorted_buffer[self.sample_minibatch[i][1]][1])
#        
#            k=0
##            find the sample to do the permutation
#            while k<len(self.sorted_buffer) and not self.sorted_buffer[k] == self.sample_minibatch[i]:
#                k+=1
#            
#            if k<len(self.sorted_buffer):
#                self.td_error.pop(k) 
#                self.sorted_buffer.pop(k)  
#            
#           
    def sort_buffer(self):
#        l = zip(self.td_error,self.sorted_buffer)
#        l.sort(reverse=True)
#        self.td_error=[e[0] for e in l]
#        self.sorted_buffer=[e[1] for e in l]
        #sorted(self.sorted_buffer, key=lambda s:s[1].reward, reverse=True)
        self.sorted_buffer.sort(reverse=True) 
#        print self.sorted_buffer[0]
#        print self.sorted_buffer[-1]
#        print self.distribution  
#        print self.dist_sur_k 

