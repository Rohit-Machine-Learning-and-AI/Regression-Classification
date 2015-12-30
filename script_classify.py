import csv
import random
import math
import numpy as np
import time
import classalgorithms_Copy as algs
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
 
def splitdataset(dataset, trainsize=1000, testsize=800, testfile=None):
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    numinputs = dataset.shape[1]-1
    Xtrain = dataset[randindices[0:trainsize],0:numinputs]
    ytrain = dataset[randindices[0:trainsize],numinputs]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],0:numinputs]
    ytest = dataset[randindices[trainsize:trainsize+testsize],numinputs]

    if testfile is not None:
        testdataset = loadcsv(testfile)
        Xtest = dataset[:,0:numinputs]
        ytest = dataset[:,numinputs]        
        
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))
 
 
def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def loadsusy():
#full path given    
    dataset = np.genfromtxt('G:\\1MS\Course Material\ML\\6. Assignment 3\\a3barebones\susyall.csv', delimiter=',')
    trainset, testset = splitdataset(dataset)    
    return trainset,testset

if __name__ == '__main__':
    trainset, testset = loadsusy()
    
    print('\nTrain={0} and Test={1} samples').format(trainset[0].shape[0], testset[0].shape[0])
           
    nnparams = {'ni': trainset[0].shape[1], 'nh': 64, 'no': 1}
    classalgs = {
                #'Random': algs.Classifier(),
                 #'Linear Regression': algs.LinearRegressionClass(),
                 #'Naive Bayes': algs.NaiveBayes({'usecolumnones': False}),
                 #'Naive Bayes Ones': algs.NaiveBayes(),
                 'Neural Network': algs.NeuralNet(nnparams),
                 #'Logistic Regression': algs.LogitReg()
                 }
    
    alg = algs.LogitReg()
    #alg = LogisticRegression(random_state = 1)
    scores = cross_validation.cross_val_score\
        (alg, trainset[0], trainset[1], cv=3, scoring="accuracy")
    print (scores.mean())
    
    #kf = KFold(trainset[0].shape[0], n_folds=3, random_state=1)
    #predictions = []
    #for train, test in kf:
    #    print("{},{}".format(train,test))
    #    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    #    train_predictors = (trainset[0].iloc[train,:])
    #    # The target we're using to train the algorithm.
    #    train_target = trainset[1].iloc[train]
    #    # Training the algorithm using the predictors and target.
    #    alg.fit(train_predictors, train_target)
    #    # We can now make predictions on the test fold
    #    test_predictions = alg.predict(trainset[0].iloc[test,:])
    #    predictions.append(test_predictions)
    #print predictions

    
    #print("\n>> Cumulative Time taken (in seconds):")
    #accuracy = {}  
    #
    #for learnername, learner in classalgs.iteritems():
    #    learner.fit(trainset[0], trainset[1])
    #    predictions = learner.predict(testset[0])
    #    accuracy[learnername] = getaccuracy(testset[1], predictions)
    #    
    #print '\n>> Accuracies (In descending order):'
    #    #below sorting code is from stackoverflow
    #accuracy_view = [ (v,k) for k,v in accuracy.iteritems() ]
    #accuracy_view.sort(reverse=True)
    #for v,k in accuracy_view:
    #    print "%s: %d" % (k,v)
        
 
