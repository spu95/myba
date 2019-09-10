'''
Created on 01.09.2019

@author: s. puglisi
@summary: experiment file for the dual l1/l2-soft-svm
'''

import numpy as np
import exputils

# Parameters for the experiment
tol = 1e-10
C = 1
max_it = 1500
categories = ['rec.autos', 'rec.motorcycles']
# categories = ['sci.electronics', 'comp.sys.ibm.pc.hardware']
       
def plot_fig1(title1, title2, ax1, ax2, num_analysis):
    # adjust objective value
    obj_val = np.array(num_analysis['obj_val'])+abs(num_analysis['obj_val'][-1])
    ax1.plot(num_analysis['iteration'], obj_val, '-k')
    ax1.set_title(title1)
    ax1.set_yscale('log')
    ax1.set_ylim(0,1e1)
    ax1.set_ylabel(r'$f(\alpha^k)$')

    
    ax2.plot(num_analysis['iteration'], num_analysis['proj_grad'],'-k')
    ax2.set_title(title2)
    axs[0,1].set_yscale('log')
    ax2.set_ylim(tol/10,1e3)
    ax2.set_ylabel(r'$\Delta(\alpha^k)$')

def plot_fig2(title1, ax1, num_analysis):
    Y = np.array(num_analysis['obj_val'])
    Y = Y + abs(Y[-1])
    Y = np.divide(Y[1:-1],Y[0:-2]) 
    ax1.plot(num_analysis['iteration'][1:-1], Y, '-k')
    ax1.set_title(title1)
    ax1.set_ylim(0,1.1)

num_analysis1 = { 'iteration' : [], 'obj_val' : [], 'proj_grad' : [] }
num_analysis2 = { 'iteration' : [], 'obj_val' : [], 'proj_grad' : [] }

# get data from scikit-learn dataset
print('setup experiment...', categories)
X,X_test,twenty_train,twenty_test = exputils.setup_data(categories)

# inject vectors into R^{n+1} by inj(x) := (x,1) 
X = exputils.inject_vecs(X)
X_test = exputils.inject_vecs(X_test)

print('train models...')

# train models and run some tests    
print('train l1-soft-svm model...')
from svm import DualCoordinateDecentSVM
clfL1SVM = DualCoordinateDecentSVM(tol=tol,C=C,shuffle=True,
    verbose=1,num_analysis=num_analysis1,loss='hinge',
    max_it=max_it)
clfL1SVM.fit(X, twenty_train.target)
score = clfL1SVM.score(X_test, twenty_test.target)
print('test score for l1-soft-svm: %f.' % score)

print('train l2-soft-svm model...')
clfL2SVM = DualCoordinateDecentSVM(tol=tol,C=C,shuffle=True,
    verbose=1,num_analysis=num_analysis2,loss='squared_hinge',
    max_it=max_it)
clfL2SVM.fit(X, twenty_train.target)
score = clfL2SVM.score(X_test, twenty_test.target)
print('test score for l2-soft-svm: %f.' % score)

# create plots for objective vals and stopping conditions
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

f1, axs = plt.subplots(2, 2, sharex='col', sharey='col')
plot_fig1('L1-Soft-SVM', 
          'L1-Soft-SVM', 
          axs[0,0], axs[0,1], num_analysis1)
plot_fig1('L2-Soft-SVM', 
          'L2-Soft-SVM', 
          axs[1,0], axs[1,1], num_analysis2)

f2, axs = plt.subplots(1, 2, sharex='col', sharey='col')
plot_fig2('L1-Soft-SVM', 
          axs[0], num_analysis1)
plot_fig2('L2-Soft-SVM',
          axs[1], num_analysis2)

exputils.adjust_figures(plt)  

plt.show()
