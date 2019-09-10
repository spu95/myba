'''
Created on 01.09.2019

@author: s. puglisi
@summary: experiment file for the dual l1/l2-soft-svm
'''

import numpy as np
import exputils

# Parameters
tol = 1e-10
C = 1
categories = ['sci.electronics', 'comp.sys.ibm.pc.hardware']
max_it = 2500
print('experiment 2: parameters (C,tol,max_it,categories)', C, tol, max_it, categories)

def plot_fig1(title1, title2, ax1, ax2, num_analysis, offset):
    # adjust objective value
    obj_val = np.array(num_analysis['obj_val'])+offset
    ax1.plot(num_analysis['iteration'], obj_val, '-k')
    ax1.set_title(title1)
    ax1.set_yscale('log')
    ax1.set_ylim(0,1e1)
    ax1.set_ylabel(r'$f(\alpha^k)$ with offset %2.5f' % offset)
    
    ax2.plot(num_analysis['iteration'], num_analysis['proj_grad'],'-k')
    ax2.set_title(title2)
    axs[0,1].set_yscale('log')
    ax2.set_ylabel(r'$\Delta(\alpha^k)$')
    ax2.set_ylim(tol/10,1e3)

# get data from scikit-learn dataset
print('setup experiment...')
X,X_test,twenty_train,twenty_test = exputils.setup_data(categories)

# inject vectors into R^{n+1} by inj(x) := (x,1) 
X = exputils.inject_vecs(X)
X_test = exputils.inject_vecs(X_test)

print('train models...')

num_analysis1 = { 'iteration' : [], 'obj_val' : [], 'proj_grad' : [] }
num_analysis2 = { 'iteration' : [], 'obj_val' : [], 'proj_grad' : [] }

# train models and run some tests    
print('train l2-soft-svm model with shuffle strategy...')
from svm import DualCoordinateDecentSVM
clfL1SVM = DualCoordinateDecentSVM(tol=tol,C=C,shuffle=True,
    verbose=1,num_analysis=num_analysis1,
    loss='squared_hinge',max_it=max_it)
clfL1SVM.fit(X, twenty_train.target)

print('train l2-soft-svm model without shuffle strategy...')
clfL2SVM = DualCoordinateDecentSVM(tol=tol,C=C,shuffle=False,
    verbose=1,num_analysis=num_analysis2,
    loss='squared_hinge',max_it=max_it)
clfL2SVM.fit(X, twenty_train.target)

# create plots for objective vals and stopping conditions
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
plt.rcParams['axes.grid'] = True
offset = -num_analysis1['obj_val'][-1]
f1, axs = plt.subplots(2, 2, sharex='col', sharey='col')
plot_fig1('(with shuffle)', 
          '(with shuffle)', 
          axs[0,0], axs[0,1], num_analysis1,offset)
plot_fig1('(without shuffle)', 
          '(without shuffle)', 
          axs[1,0], axs[1,1], num_analysis2,offset)

exputils.adjust_figures(plt)  

plt.show()
