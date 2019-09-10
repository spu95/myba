'''
Created on 01.09.2019

@author: s. puglisi
@summary: experiment file for the dual l1/l2-soft-svm
'''
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

inf = float("inf")
minf = float("-inf")

'''
@summary: A coordinate descent algorithm for
dual L1-loss- and L2-loss SVM problems. 
Python implementation of the function solve_l2r_l2_svc 
in Liblinear, Hsieh et al., ICML 2008

@see https://github.com/cjlin1/liblinear for the
original c/c++ implementation
'''
def solve_l2r_l2_svc(X,y,C=1.,tol=1e-8,max_it=1500,shuffle=True,
         loss='squared_hinge',verbose=0,num_analysis=None):
    length = len(X) # sample size
    dim = X.shape[1]# dimension of samples

    a = np.zeros(length)
    Q = np.zeros(length)
    w = np.zeros(dim)
    Dii = 0.
    
    if loss == 'hinge':
        U = C
    elif loss == 'squared_hinge':
        U = inf
    else:
        raise ValueError('loss must be either hinge or squared_hinge')
    
    PG = 0
    M,m = -inf,inf
    k,j = 0,0 # outer iterations, all iterations
    
    # prepeare data
    if loss == 'squared_hinge':        
        for i in range(0, length):
            Q[i] += 0.5/C
        Dii = 0.5/C
    for i in range(0, length):
        Q[i] = X[i].dot(X[i])
    it = np.arange(length, dtype='int32')
        
    while(k < max_it):
        M,m = -inf,inf
       
        if shuffle: np.random.shuffle(it[:-1])
        s = 0
        while s < length:
            i = it[s]
            yi = y[i]
            
            # calculate new direction
            G = yi*w.dot(X[i])-1.+Dii*a[i]
            
            # project new direction
            PG = 0.
            if a[i] == 0.:
                if G < 0.:
                    PG = G
            elif a[i] == U:
                if G > 0.:
                    PG = G
            else:
                PG = G

            # update a[i] and w
            if abs(PG) > 0.:
                old = a[i]
                a[i] = min(max(old - G/Q[i], 0.), U)
                w += (a[i]-old)*yi*X[i]
                
            # calculate termination condition
            m = min(PG,m)
            M = max(PG,M)

            s += 1
            j += 1   
        k += 1
        
        # add analysis information
        if num_analysis is not None: 
            # obj_val = 0.5 * ((a^t)Qa + (a^t)Da) - e^t*a
            v = 0.5 * (w.dot(w) + a.dot(a) * Dii) - np.sum(a) 
            num_analysis['iteration'].append(j) 
            num_analysis['obj_val'].append(v)
            num_analysis['proj_grad'].append(M-m)

        # check termination condition
        if M-m < tol:
            break
    
    # some extra informations        
    if verbose == 1:
        # obj_val = 0.5 * ((a^t)Qa + (a^t)Da) - e^t*a
        v = 0.5 * (w.dot(w) + a.dot(a) * Dii) - np.sum(a)
        nSV = np.count_nonzero(a)
        print("\nM-m=%5.3e M: %5.3e m=%5.3e - Steps: k=%d,%d" 
              % (M-m, M, m, k, j))
        print("Objective value = %.12f" % (v))
        print("nSV = %d" % nSV)

    return w, k

class DualCoordinateDecentSVM(BaseEstimator, ClassifierMixin):
    def __init__(self,C=1,tol=1e-4,loss='squared_hinge',
            max_it=1500,shuffle=True,verbose=0,num_analysis=None):
        self.tol = tol
        self.loss = loss
        self.shuffle = shuffle
        self.verbose = verbose
        self.coef_ = 0
        self.num_analysis = num_analysis
        self.max_it = max_it
        self.C = C
        
    def fit(self,X,y):
        self.coef_, self.n_iter_ = solve_l2r_l2_svc(X,y,C = self.C,
                tol=self.tol,loss=self.loss,shuffle=self.shuffle,
                max_it = self.max_it,verbose=self.verbose,
                num_analysis=self.num_analysis)
        return self

    def predict(self,X):
        y = np.sign(X.dot(self.coef_))
        return y
        
    def score(self,X,y):
        if self.coef_ is None:
            raise RuntimeError('Train your model first')
        else:
            b = self.predict(X)*y
            scores = len(b[b>0])
            return scores /len(X)
