# -*- coding: utf-8 -*-
"""

@author: sigaud
"""
import random
from collections import deque
from minibatch import minibatch
from sample import sample
import numpy as np
 
class replay_buffer(object):
    """
    A sample from an agent environment interaction
    """
    def __init__(self, min_filled,size):
        self.buffer = deque([])
        self.bests = deque([])
        self.sample_minibatch=[]
        self.size = size
        self.bests_size = size
        self.min_filled = min_filled
        self.reward_min = float("Inf")
        self.reward_max = -float("Inf")
        
        self.temporal_buffer_size=size;
        
        self.temporal_buffer=[];
        self.alpha=1
        
#        self.mean=1
#        self.std=0.1
        
        self.distribution=[(1.0/i)**self.alpha for i in range(1,size+1)]
        self.distribution=[sum(self.distribution[k*size/64:(k+1)*size/64]) for k in range(64)]
        s = sum(self.distribution)
        self.distribution=[i/s for i in self.distribution]
        
    def updatealpha(self,alpha):
        if abs(self.alpha-alpha)< 0.1:
            return
        self.alpha=alpha  
        self.distribution=[(1.0/i)**self.alpha for i in range(1,self.size+1)]
        self.distribution=[sum(self.distribution[k*self.size/64:(k+1)*self.size/64]) for k in range(64)]
        s = sum(self.distribution)
        self.distribution=[i/s for i in self.distribution]
        
    def flush(self):
        self.buffer = deque([])
            
    def current_size(self):
        return len(self.buffer)
        
    def store_samples_from_batch(self, s_t, a_t, r_t, s_t_next):
        assert(False)
#        self.reward_max = max(r_t, self.reward_max)
#        self.reward_min = min(r_t, self.reward_min)   
#        print self.reward_min, self.reward_max
#        for i in range(len(s_t)):
#            if (self.isFull()):
#                self.buffer[random.randint(self.size/5, self.size-1)] = [list(s_t[i]), list(a_t[i]), r_t[i], list(s_t_next[i])]
#            else:
#                self.buffer.append([list(s_t[i]), list(a_t[i]), r_t[i], list(s_t_next[i])])
    
    def flush_temporal_buffer(self):
        self.temporal_buffer=[]
#
#    def update_temporal_buffer(self):
#        error = 0.01
#        for i in range(len(self.temporal_buffer)):
##           print self.temporal_buffer_size, self.current_size()-1
#           self.buffer[random.randint(0,1+min(self.temporal_buffer_size, self.current_size()/10-1))] = ((error,self.temporal_buffer[i]))
#         
    
    def update_bests_buffer(self): 
        error = 0.01
        for i in range(len(self.temporal_buffer)):
#           print self.temporal_buffer_size, self.current_size()-1
            if len(self.bests)>self.bests_size: 
                self.bests[min(random.randint(len( self.bests)/10,1+len( self.bests)),len(self.bests)-1)] =   (error,self.temporal_buffer[i] ) 
            else:
                self.bests.append( (error,self.temporal_buffer[i] ) )
    
    def store_one_sample(self, sample):
        self.reward_max = max(sample.reward, self.reward_max)
        self.reward_min = min(sample.reward, self.reward_min)   
#        if len(self.bests)==0 or self.bests[-1].reward < sample.reward:
#            i=1
#            while i<=len(self.bests) and self.bests[-i].reward < sample.reward:
#                i+=1
#            self.bests.insert(len(self.bests)-i, sample)
#            if len(self.bests)>1000:
#                self.bests.pop()
#        error = max(0,np.random.normal(loc  = self.mean,scale  =self.std) )
        self.temporal_buffer.append(sample)

        if (self.isFull()):
            # replace an older sample, but protecting the beginning
           #np.random.uniform(self.buffer[self.current_size()/5][0],self.buffer[self.current_size()/2][0])
           error = self.buffer[self.size/50][0]
           self.buffer[random.randint(self.size/10, self.size-1)] = ((error,sample))
#            self.buffer.appendleft((error,sample))
#            self.buffer.pop()
#        elif len( self.buffer)<5:
#            self.buffer.append((1,sample))
        else:
#            print self.buffer[0][0],self.buffer[self.current_size()/5][0]
#            error = 1#np.random.uniform(self.buffer[self.current_size()/5][0],self.buffer[self.current_size()/2][0])
            self.buffer.append((0.01,sample))
            
        if  len(self.bests)<5:
            self.bests.append((0.01,sample))
#    def store_one_sample(self, sample):
#        self.reward_max = max(sample.reward, self.reward_max)
#        self.reward_min = min(sample.reward, self.reward_min)   
#        if len(self.bests)==0 or self.bests[-1].reward < sample.reward:
#            i=1
#            while i<=len(self.bests) and self.bests[-i].reward < sample.reward:
#                i+=1
#            self.bests.insert(len(self.bests)-i, sample)
#            if len(self.bests)>1000:
#                self.bests.pop()
##        error = max(0,np.random.normal(loc  = self.mean,scale  =self.std) )
#        self.temporal_buffer.append(sample)
#
#        if (self.isFull()):
#            # replace an older sample, but protecting the beginning
#           #np.random.uniform(self.buffer[self.current_size()/5][0],self.buffer[self.current_size()/2][0])
#           error = self.buffer[self.temporal_buffer_size][0]
#           self.buffer[random.randint(self.size/10, self.size-1)] = ((error,sample))
##            self.buffer.appendleft((error,sample))
##            self.buffer.pop()
##        elif len( self.buffer)<5:
##            self.buffer.append((1,sample))
#        else:
##            print self.buffer[0][0],self.buffer[self.current_size()/5][0]
##            error = 1#np.random.uniform(self.buffer[self.current_size()/5][0],self.buffer[self.current_size()/2][0])
#            self.buffer.append((0.01,sample))
#            

        
    def isFullEnough(self):
        return (self.current_size()>self.min_filled)

    def isFull(self):
        return (self.current_size()>=self.size)

    def get_state(self,index):
        return self.buffer[index][1].state

    def get_random_minibatch(self,batch_size):
            states = []
            rewards = []
            actions = []
            next_states = []
            for i in range(batch_size):
                if random.uniform(0.0,1.0)<0.1:
                    index= random.randint(0, len(self.bests)-1)
                    sample = self.bests[index]
                else:
                    index= random.randint(0, self.current_size()-1)
                    sample = self.buffer[index][1]
                    
                states.append(sample.state) #no need to put into [] because it is already a vector
                actions.append(sample.action) #no need to put into [] because it is already a vector
                if self.reward_max-self.reward_min == 0:
                    rewards.append([sample.reward])
                else:                
                    rewards.append([(sample.reward-self.reward_min)/(self.reward_max-self.reward_min)*2.0-1.0])
                #print((sample.reward-self.reward_min)/(self.reward_max-self.reward_min)*2.0-1.0)
                next_states.append(sample.next_state) #no need to put into [] because it is already a vector
            return minibatch(states,actions,rewards,next_states)
   
    def get_td_error_sorted_minibatch(self,batch_size):
            states = []
            rewards = []
            actions = []
            next_states = []
            self.sample_minibatch=[]
            
                        
            
            segment_index= np.random.choice(range( batch_size ), p=self.distribution)
            inf1 = segment_index*len(self.bests)/batch_size
            sup1 = 1+(segment_index+1)*len(self.bests)/batch_size
            inf2 = segment_index*len(self.buffer)/batch_size
            sup2 = 1+(segment_index+1)*len(self.buffer)/batch_size
            for i in range(batch_size):
                if random.uniform(0.0,1.0)<0.7:
#                    index= random.randint(0, len(self.bests)-1)
#                    sample = self.bests[index] 
#                    segment_index= np.random.choice(range( batch_size ), p=self.distribution)
                    index = np.random.randint( inf1,sup1)
                    index = min(len(self.bests)-1,max(0,index))
                    sample = self.bests[index][1]
                    self.sample_minibatch.append((0,index))
                else:
#                    index= random.randint(0, self.current_size()-1)
#                    segment_index= np.random.choice(range( batch_size ), p=self.distribution)
                    index = np.random.randint( inf2,sup2)
                    index = min(self.current_size()-1,max(0,index))
                    sample = self.buffer[index][1]
                    self.sample_minibatch.append((1,index))
                
                
                states.append(sample.state) #no need to put into [] because it is already a vector
                actions.append(sample.action) #no need to put into [] because it is already a vector
#                rewards.append([sample.reward])
                if self.reward_max-self.reward_min == 0:
                    rewards.append([sample.reward])
                else:                
                    rewards.append([self.getNormalReward(  sample.reward)])
                #print((sample.reward-self.reward_min)/(self.reward_max-self.reward_min)*2.0-1.0)
                next_states.append(sample.next_state) #no need to put into [] because it is already a vector
            return minibatch(states,actions,rewards,next_states)
    def getNormalReward(self, reward):
        return  (reward-self.reward_min)/(self.reward_max-self.reward_min)*2.0-1.0

    def sort_buffer_by_td_error(self): 
        self.buffer=  deque(sorted(self.buffer,reverse=True))
        self.bests=  deque(sorted(self.bests,reverse=True))
#        index = min(self.temporal_buffer_size,self.current_size()/40)
#        bf =  list(self.buffer) 
#        head = bf[0:index]
#        tail = bf[index:]
        
#        tail = sorted(tail, key=lambda x:(x[0], x[1].reward))
#        head = sorted(head,reverse = True )
#        tail = sorted(tail,reverse = True )
#        self.buffer=deque(head+tail)
#        self.buffer=sorted(self.buffer,reverse = True )
        
#        td_error = [x[0] for x in self.buffer]
#        self.mean = np.mean(td_error)
#        self.std = np.std(td_error)
#        self.updatealpha(self.buffer[0][0])
        
#        self.buffer=  sorted(self.buffer ,reverse=True,key=lambda x:(x[0], x[1].reward))
       
    def showConvergence(self):
        index =  self.current_size()/40 
        print "\nalpha=",self.alpha
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (0, self.buffer[0][0], self.buffer[0][1].state[0],self.buffer[0][1].state[1], self.buffer[0][1].action, self.buffer[0][1].next_state[0],self.buffer[0][1].next_state[1], self.getNormalReward(self.buffer[0][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (5, self.buffer[5][0], self.buffer[5][1].state[0],self.buffer[5][1].state[1], self.buffer[0][1].action, self.buffer[5][1].next_state[0],self.buffer[5][1].next_state[1], self.getNormalReward(self.buffer[5][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (10, self.buffer[10][0], self.buffer[10][1].state[0],self.buffer[10][1].state[1], self.buffer[10][1].action, self.buffer[10][1].next_state[0],self.buffer[10][1].next_state[1], self.getNormalReward(self.buffer[10][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (50, self.buffer[50][0], self.buffer[50][1].state[0],self.buffer[50][1].state[1], self.buffer[50][1].action, self.buffer[50][1].next_state[0],self.buffer[50][1].next_state[1], self.getNormalReward(self.buffer[50][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (100, self.buffer[100][0], self.buffer[100][1].state[0],self.buffer[100][1].state[1], self.buffer[100][1].action, self.buffer[100][1].next_state[0],self.buffer[100][1].next_state[1], self.getNormalReward(self.buffer[100][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (index, self.buffer[index][0],self.buffer[index][1].state[0],self.buffer[index][1].state[1],self.buffer[index][1].action, self.buffer[index][1].next_state[0],self.buffer[index][1].next_state[1],self.getNormalReward(self.buffer[index][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (index+5, self.buffer[index+5][0],self.buffer[index+5][1].state[0],self.buffer[index+5][1].state[1],self.buffer[index+5][1].action, self.buffer[index+5][1].next_state[0],self.buffer[index+5][1].next_state[1],self.getNormalReward(self.buffer[index+5][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (index+10, self.buffer[index+10][0],self.buffer[index+10][1].state[0],self.buffer[index+10][1].state[1],self.buffer[index+10][1].action, self.buffer[index+10][1].next_state[0],self.buffer[index+10][1].next_state[1],self.getNormalReward(self.buffer[index+10][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (index+100, self.buffer[index+100][0],self.buffer[index+100][1].state[0],self.buffer[index+100][1].state[1],self.buffer[index+100][1].action, self.buffer[index+100][1].next_state[0],self.buffer[index+100][1].next_state[1],self.getNormalReward(self.buffer[index+100][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (index*12/10,self.buffer[index*12/10][0],  self.buffer[index*12/10][1].state[0],self.buffer[index*12/10][1].state[1],  self.buffer[index*12/10][1].action, self.buffer[index*12/10][1].next_state[0],self.buffer[index*12/10][1].next_state[1],self.getNormalReward(self.buffer[index*12/10][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (index*15/10, self.buffer[index*15/10][0],self.buffer[index*15/10][1].state[0],self.buffer[index*15/10][1].state[1],self.buffer[index*15/10][1].action, self.buffer[index*15/10][1].next_state[0],self.buffer[index*15/10][1].next_state[1],self.getNormalReward(self.buffer[index*15/10][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (index*18/10,self.buffer[index*18/10][0],  self.buffer[index*18/10][1].state[0],self.buffer[index*18/10][1].state[1],  self.buffer[index*18/10][1].action, self.buffer[index*18/10][1].next_state[0],self.buffer[index*18/10][1].next_state[1],self.getNormalReward(self.buffer[index*18/10][1].reward))

        print "\ni=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (0, self.bests[0][0], self.bests[0][1].state[0],self.bests[0][1].state[1], self.bests[0][1].action, self.bests[0][1].next_state[0],self.bests[0][1].next_state[1], self.getNormalReward(self.bests[0][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (5, self.bests[5][0], self.bests[5][1].state[0],self.bests[5][1].state[1], self.bests[0][1].action, self.bests[5][1].next_state[0],self.bests[5][1].next_state[1], self.getNormalReward(self.bests[5][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (10, self.bests[10][0], self.bests[10][1].state[0],self.bests[10][1].state[1], self.bests[10][1].action, self.bests[10][1].next_state[0],self.bests[10][1].next_state[1], self.getNormalReward(self.bests[10][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (50, self.bests[50][0], self.bests[50][1].state[0],self.bests[50][1].state[1], self.bests[50][1].action, self.bests[50][1].next_state[0],self.bests[50][1].next_state[1], self.getNormalReward(self.bests[50][1].reward))
        print "i=%d:\ttd_error=%.6f\tstate=[%.6f,%.6f]\taction=%.6f \tnext_state=[%.6f,%.6f] \treward=%.6f"% (100, self.bests[100][0], self.bests[100][1].state[0],self.bests[100][1].state[1], self.bests[100][1].action, self.bests[100][1].next_state[0],self.bests[100][1].next_state[1], self.getNormalReward(self.bests[100][1].reward))
    def sort_buffer_by_reward(self):
        self.buffer=  deque(sorted(self.buffer,reverse=True,key=lambda x:x[1].reward))
        print self.buffer[0][1].reward,self.buffer[self.current_size()/10][1].reward
                         
    def update_td_error(self,td_err): 
       for i in range(len(td_err)):
           if self.sample_minibatch[i][0]==1:
               sample =  self.buffer[self.sample_minibatch[i][1]][1]
               error = td_err[i][0]
#               if abs(td_err[i][0]) > self.mean+ 2* self.std :
#                   error = 1 
#               elif abs(td_err[i][0])>  self.mean -2* self.std:
#                   error =   self.mean
#               else:
#                   error = 0
               self.buffer[self.sample_minibatch[i][1]]=(error,sample)
           else:
               sample =  self.bests[self.sample_minibatch[i][1]][1]
               error = td_err[i][0] 
               self.bests[self.sample_minibatch[i][1]]=(error,sample) 