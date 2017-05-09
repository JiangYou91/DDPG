# -*- coding: utf-8 -*-
"""
Created on Mon May 08 23:42:43 2017

@author: you
"""
#import numpy as np
#
#class sampleTree:
#    import numpy as np
    
#    class binarytree:
#        self.left=None
#        self.right=None
#        self.sample=None
#        self.td_error=None 
#        self.size=1
#        
#        def __init__(self,td_err,sample):
#            self.sample=sample
#            self.td_error=td_err
#        def pop_left(self):
#            tmp= self.left
#            self.left=None
#            if tmp!=None:
#                 self.size-=1
#            return tmp
#        def pop_right(self):
#            tmp= self.right
#            self.right=None
#            if tmp!=None:
#                 self.size-=1
#            return tmp
#        def pop_min(self):
#            self.size-=1
#            child=self.right
#            if child.right!=None:
#                child= child.pop_min()
#            else: 
#                self.right=None
#            return child
#        def pop_max(self):
#            self.size-=1
#            child=self.left 
#            if child.left!=None:
#                child= child.pop_max()
#            else: 
#                self.left=None
#            return child
#        def put(self,td_err,sample):
#            self.size+=1
#            if self.td_error>td_err:
#                if self.right==None:
#                    self.right= self(td_err,sample)
#                else:
#                    self.right.put(td_err,sample)
#            else:
#                if self.left==None:
#                    self.left= self(td_err,sample)
#                else:
#                    self.left.put(td_err,sample)
#        
#        def pop_random(self):
##                i= np.random.choice(["self","left","right"])
#            if self.left==None and self.right==None:
#                return self, 0, None
#            elif self.left!=None and self.right==None:
#                i = np.random.randint(0,2)
#                if i==0:
#                    return self, 0, self.pop_left()
#                else:
#                    self.size-=1
#                    tmp,code,replace = self.left.pop_random()
#                    if code==0:
#                        self.left = replace
#                    return tmp, 1, None
#            elif self.right!=None and self.left==None:
#                i = np.random.randint(0,2)
#                if i==0: 
#                    return self, 0, self.pop_right()
#                else:
#                    self.size-=1 
#                    tmp,code,replace = self.right.pop_random()
#                    if code==0:
#                        self.right = replace
#                    return tmp, 1, None
#            else: 
##                i= np.random.choice(["self","left","right"])
#                i = np.random.randint(0,3)
#                if i==0: 
#                    return self, 0, self.merge(self.left,self.right)
#                             
#                elif i==1:
#                    self.size-=1
#                    tmp,code,replace = self.left.pop_random()
#                    if code==0:
#                        self.left = replace
#                    return tmp, 1, None
#                else:
#                    self.size-=1 
#                    tmp,code,replace = self.right.pop_random()
#                    if code==0:
#                        self.right = replace
#                    return tmp, 1, None
#                       
#        def merge(self, t1, t2):
#            if t1.td_error<t2.td_error:
#                tmp=t1
#                t1=t2
#                t2=tmp
#            
##              t1.td_error>t2.td_error:
#            if t1.right==None:
#                t1.right=t2
#                return t1
#            elif t2.left==None:
#                t2.left=t1
#                return t2
#            else:
#                if np.random.randint(0,2)==0:
#                    t1.right=merge( t1.right, t2)
#                    return t1
#                else:
#                    t2.left=merge( t2.left, t1)
#                    return t2 
   
class sampleTree:
    import numpy as np  
    class td_error_sample:  
        def __init__(self,td_error,sample): 
            self.td_error=td_error
            self.sample=sample
        def __str__(self):
            return "("+str(self.td_error)+"," +str(self.sample)+")"
    class child:
        import heapq 
        def __init__(self,size): 
            self.max=0
            self.min=0
            self.size = size 
            self.h=[]
            heapq._heapify_max(h)
        def put(self,e):
            self.max=max(self.max,e.td_error)
            self.min=min(self.min,e.td_error)
            heapq.heappush(self.h,e)
        def get_N_largest(self,N):
            return  heapq.nlargest(N,self.h , key=lambda x:x.td_error)
        def pop_N_smallest(self,N):
            return  heapq.nsmallest(N,self.h , key=lambda x:x.td_error)
        def getCurrentSize(self):
            return len(self.h)
        
    def __init__(self,N=80,size=40000, alpha=0.8): 
        self.size=0
        self.childrens=[]
        
        self.alpha 
        self.distribution = []    
        
        self.alpha=alpha
        self.size=size 
        self.distribution=[(1.0/i)**self.alpha for i in range(1,size+1)]
        self.distribution=[sum(self.distribution[k*size/N:(k+1)*size/N]) for k in range(N)]
        s = sum(self.distribution)
        self.distribution=[i/s for i in self.distribution]
        
    def get(self,index):
        assert index<self.N
        return  
    def put(self,td_err,sample):
        if len(self.childrens)<self.size:
            self.childrens.append(self.binarytree(td_err,sample))
            self.childrens.sort(reverse=True,key= lambda x:x.td_error)
        else:
            i=0
            while i<len(self.childrens)-1 and td_err<self.childrens[i].td_error :
                i+=1
            self.childrens[i].put(td_err,sample)
            
            
test =  sampleTree(64)




import heapq
import numpy as np
a=zip(np.random.random_sample(100),range(100))

a
h =[]
heapq._heapify_max(h)
for i in a:
    heapq.heappush(h,td_error_sample(i[0],i[1]))

heapq._heapify_max(h)
for i in a:
    heapq.heappush(h,i)
heapq.heappop(h)
b = heapq.nlargest(64, h, key=lambda x:x.td_error)

for i in b:
    i.td_error=0
b
for i in b:
    print  i.td_error
for i in h:
    print  i.td_error

b[0:1]

class sampleTree:
    import numpy as np  
    import heapq
    class td_error_sample:  
        def __init__(self,td_error,sample): 
            self.td_error=td_error
            self.sample=sample
        def __str__(self):
            return "("+str(self.td_error)+"," +str(self.sample)+")"
     
        
    def __init__(self,N=80,size=40000, alpha=0.8): 
        self.size=0
        self.h=[]
        
        self.alpha 
        self.distribution = []    
        
        self.alpha=alpha
        self.size=size  
        
        self.distribution=[(1.0/i)**self.alpha for i in range(1,size+1)]
        self.distribution=[sum(self.distribution[k*size/N:(k+1)*size/N]) for k in range(N)]
        s = sum(self.distribution)
        self.distribution=[i/s for i in self.distribution]
       
    def getCurrentSize(self):
        return len(self.h) 
    def get_N_largest(self,index,N):
        assert index<self.N 
        inf=
        sup=
        b = heapq.nlargest(64, h[self.getCurrentSize()], key=lambda x:x.td_error)
        getCurrentSize(self)
        return  
    def put(self,td_err,sample):
        if len(self.childrens)<self.size:
            self.childrens.append(self.binarytree(td_err,sample))
            self.childrens.sort(reverse=True,key= lambda x:x.td_error)
        else:
            i=0
            while i<len(self.childrens)-1 and td_err<self.childrens[i].td_error :
                i+=1
            self.childrens[i].put(td_err,sample)
            
            