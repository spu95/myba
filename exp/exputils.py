'''
Created on 03.09.2019

@author: spuglisi
'''
import numpy as np

def setup_data(categories, dense = True):
    '''
    useful docs for the following code:
    https://scikit-learn.org/stable/tutorial/
    text_analytics/working_with_text_data.html
    '''
    
    # get data from scikit-learn dataset
    from sklearn.datasets import fetch_20newsgroups
    twenty_train = fetch_20newsgroups(subset='train',
        categories=categories, shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(subset='test',
        categories=categories, shuffle=True, random_state=42)
    
    # transform data into the VSM (Vector Space Model)
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    count_vect.fit(twenty_train.data + twenty_test.data)
    
    if dense:
        X = np.asarray(count_vect.transform(twenty_train.data).todense())
        X_test = np.asarray(count_vect.transform(twenty_test.data).todense())
    else:
        X = count_vect.transform(twenty_train.data)
        X_test = count_vect.transform(twenty_test.data)
    
    # adjust target
    twenty_train.target[twenty_train.target == 0] = -1
    twenty_test.target[twenty_test.target == 0] = -1
    
    return X,X_test,twenty_train,twenty_test

def inject_vecs(X):
    '''
    inject vectors into R^{n+1} by inj(x) := (x,1) 
    '''
    X = np.c_[X, np.ones(X.shape[0])]
    return X

def adjust_figures(plt):
    figs = list(map(plt.figure, plt.get_fignums()))
    for f in figs:
        for ax in f.axes:
            ax.tick_params(
                axis='x', which='both',       
                bottom=True, top=False,
                labelbottom=True)   
            ax.set_xlabel('k') 


