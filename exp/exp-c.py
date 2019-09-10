'''
Created on 02.09.2019

@author: spuglisi
'''

# get data from scikit-learn dataset
from sklearn.svm.classes import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from time import time

import exputils

print('setup experiment')

categories = [
             'alt.atheism',
            'comp.graphics',  
 'comp.os.ms-windows.misc', 
'comp.sys.ibm.pc.hardware', 
   'comp.sys.mac.hardware',
          'comp.windows.x',   
            'misc.forsale',  
               'rec.autos',
         'rec.motorcycles',
      'rec.sport.baseball',
        'rec.sport.hockey',
               'sci.crypt',
         'sci.electronics',     
                 'sci.med',      
               'sci.space',      
  'soc.religion.christian',
      'talk.politics.guns',
   'talk.politics.mideast', 
      'talk.politics.misc',
      'talk.religion.misc',
]
#categories = ['rec.autos','sci.electronics']

# get data from scikit-learn dataset
print('setup experiment...')
X,X_test,twenty_train,twenty_test = exputils.setup_data(categories, False)

machines = {
    # predict method is very slow because it does 
    # not store w (even if w is calculable)!! 
    'Lib-SVM' : SVC(kernel='linear',verbose=1), 
    'L1-Soft-SVM' : LinearSVC(loss='hinge',C=1,verbose=1),
    'L2-Soft-SVM' : LinearSVC(loss='squared_hinge',C=1,verbose=1),
    'MultinomialNB' : MultinomialNB(),
}
    
for label, machine in machines.items():
    print('train a', label, 'model...')
    t0 = time()
    machine.fit(X, twenty_train.target)  
    t1 = time()
    print("measured training time %0.3fs." % (t1 - t0))
    t0 = time()
    predicted = machine.predict(X_test)
    t1 = time()
    print("measured test time %0.3fs." % (t1 - t0))
    from sklearn import metrics
    print(metrics.classification_report(twenty_test.target, predicted,
        target_names=twenty_test.target_names))
                                        