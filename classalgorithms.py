from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
import time
import copy
import warnings
import numpy as np
from scipy import sparse
#from .externals import six
#from .utils.fixes import signature

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        
    def fit(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest
   
########################################################################                        
########################################################################                        

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, params=None ):
        self.weights = None
        if params is not None and 'regwgt' in params:
            self.regwgt = params['regwgt']
        else:
            self.regwgt = 0.01
        self.tli = time.time()   
        
        
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.inv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.regwgt*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        print("linear:"),
        print time.time() - self.tli
        
        return ytest
   
########################################################################   
#shape comments are for trainsize=1000, testsize=800 for susy               
########################################################################                        
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes; need to complete the inherited learn and predict functions """
    
    def __init__( self, params=None ):
        """ Params can contain any useful parameters for the algorithm """
        self.usecolumnones = True
        if params is not None:
            self.usecolumnones = params['usecolumnones']
        self.groupings={} 
        self.tn = time.time()   
        
            #inspired from machinelearningmastery.com
    def learn(self, Xtrain2, ytrain):
        if not self.usecolumnones:
            Xtrain = Xtrain2[:,:(len(Xtrain2[0])-2)]
        else:
            Xtrain = Xtrain2
            
        def makegroups(dataset):
            groups = [(utils.mean(attribute), utils.stdev(attribute)) for attribute in zip(*dataset)]
            return groups

        classify = {}
        classify[0] = []
        classify[1] = []
        
	for i in range(len(Xtrain)):
            vector = Xtrain[i]
            classify[ytrain[i]].append(vector)
	#print len(classify[0]) #splits Xtrain of 1000 into 2(with sum = 1000 of course)
	for Value, sample in classify.iteritems():
            self.groupings[Value] = makegroups(sample)
        #print self.groupings #gives mean and std_dev of 8 attr each for 0 and 1
	
	
    def predict(self, Xtest2):
        if not self.usecolumnones:
            Xtest = Xtest2[:,:(len(Xtest2[0])-2)]
        else:
            Xtest = Xtest2
        
        def pred(xvec):
            p = {}
            for Value, groups in self.groupings.iteritems():
            #class0: (mean1,stddev1), (mean2,stddev2)....
            #class1: (mean1,stddev1), (mean2,stddev2)....            
                p[Value] = 1
                for i in range(len(groups)):
                    mean, stdev = groups[i]
                    x = xvec[i]
                    p[Value] *= utils.calculateprob(x, mean, stdev)
            
            #only for 2 classes            
            if p[0] > p[1]: return 0
            else: return 1
            
            
        predictions = []
	for i in range(len(Xtest)):
            result = pred(Xtest[i])
            predictions.append(result)

	print("Naive Bayes "),
        if self.usecolumnones: print("with ones:"),
        else: print ("without ones:"),
        print(time.time() - self.tn)

	return predictions
	
########################################################################                        
#shape comments are for trainsize=1000, testsize=800 for susy               
########################################################################                        
        
class LogitReg(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """

    def __init__( self, params=None ):
        self.reps = 5
        self.tlo = time.time()   
        
    def fit(self, Xtrain, ytrain):
        print("We're in fit in LogitReg")
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        self.xvec = np.dot(Xtrain, self.weights)#1000x1
        self.p = utils.sigmoid(self.xvec) #1000x1
        self.P = np.diagflat(self.p) #1000x1000

        lambdaa = 0.01
        for reps in range(self.reps):
            self.weights -= (1/Xtrain.shape[0])*1*np.dot(np.dot(np.linalg.inv(np.dot(np.dot(np.dot(Xtrain.T,self.P),np.eye(np.size(Xtrain.shape[0]))-self.P),Xtrain)+lambdaa),Xtrain.T),(ytrain-self.p))
    
    
    def predict(self, Xtest):
        xvec = np.dot(Xtest, self.weights)
        ytest = utils.sigmoid(xvec)
        ym = utils.mean(ytest)
        ytest[ytest >= ym] = 1     
        ytest[ytest < ym] = 0    
 	print("logistic:"),
        print time.time() - self.tlo

        return ytest
   
########################################################################                        
#question2 - algs.LogitReg2()
#shape comments are for trainsize=1000, testsize=800 for susy               
########################################################################                        
        
class LogitReg2(Classifier):
    """ Logistic regression; need to complete the inherited learn and predict functions """

    def __init__( self, params=None ):
        self.reps = 5
        self.tlo = time.time()
        
    def learn(self, Xtrain, ytrain):        
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(Xtrain.T,Xtrain)),Xtrain.T),ytrain)
        
        #print self.xvec.shape  #1000, 
        #print ytrain.shape #1000,
        #print Xtrain.shape #1000x9
        #print self.p.shape #1000,
       
            
        lambdaa = 0.01
        for reps in range(5*self.reps):
            self.xvec = np.dot(Xtrain, self.weights)#1000x1
            self.p = 1.0 + self.xvec/math.sqrt(1.0 + np.dot(self.xvec.T, self.xvec))
            one = 1.0/math.sqrt(1.0 + np.dot(self.xvec.T, self.xvec)) #scalar
            two = Xtrain.T #9x1000
            three = 2*ytrain - self.p #1000,
            self.weights -= lambdaa * one*np.dot(two,three)
            lambdaa = lambdaa/5
            
    def predict(self, Xtest):
        xvec = np.dot(Xtest, self.weights)
        ytest = utils.sigmoid(xvec)
        ym = utils.mean(ytest)
        ytest[ytest >= ym] = 1     
        ytest[ytest < ym] = 0    
 	print("Logistic Q2:"),
        print time.time() - self.tlo

        return ytest
   

########################################################################
#question3 - algs.LogitReg3()
########################################################################                        
        
class LogitReg3(Classifier):
    '''Please use Madelon from script_classify'''
    ''' Please run one regularizer at a time, commenting others '''
    ''' In the below code, only L-infinity is uncommented'''

    def __init__( self, params=None ):
        self.reps = 5
        self.tlo = time.time()
        self.lam = 0.01   
        
    def learn(self, Xtrain, ytrain):
        inv = np.linalg.inv(np.dot(Xtrain.T,Xtrain)+self.lam)
        self.weights = np.dot(np.dot(inv,Xtrain.T),ytrain) #acc = 57
        self.weights = np.zeros(Xtrain.shape[1]) #acc = 58
        #self.weights = np.random.normal(0, 1, Xtrain.shape[1]) #acc = 50
        
        self.xvec = np.dot(Xtrain, self.weights) #2000,
        self.p = utils.sigmoid(self.xvec) #2000,
        self.P = np.diagflat(self.p) #2000x2000
        #print self.xvec.shape #2000,
        #print ytrain.shape #2000,
        #print Xtrain.shape #2000x501
        #print self.p.shape #2000,
        #print self.P.shape #2000x2000

 #       self.lam = 0.01 # because it was altered earlier
 #       for reps in range(5*self.reps):
 #           self.xvec = np.dot(Xtrain, self.weights) #2000,
 #           self.p = utils.sigmoid(self.xvec) #2000,
 #           self.P = np.diagflat(self.p) #2000x2000
 #           I = np.eye(np.size(Xtrain.shape[0]))
 #           h = np.dot(np.dot(np.dot(Xtrain.T,self.P),I-self.P),Xtrain)
 #           delta = np.dot(Xtrain.T,(ytrain-self.p))

 ##accuracy without regularizer -> 57
 #           #self.weights -= np.dot(np.linalg.inv(h),delta)
 ##accuracy with L2 regularizer -> 59
 #           self.weights -= np.dot(np.linalg.inv(h+self.lam),(delta+self.lam*self.weights))
 #           #w = w + rate * inv(h+c).(delta+lambda.w)
            
# accuracy with L1 regularizer -> 56
        #self.lam = 0.01 # because it was altered earlier
        #for reps in range(5*self.reps):
        #    delta_l1 = np.dot(Xtrain.T,(ytrain-self.p))
        #    self.weights += self.lam * (delta_l1 - np.sign(self.weights))
        #    self.lam /= 5
        
# accuracy with L3 (own regularizer) -> 60
        self.lam = 0.01 # because it was altered earlier
        for reps in range(5*self.reps):
            linf = 0
            for i in self.weights:
                ai = abs(i)
                if ai > linf: linf = ai
            delta = np.dot(Xtrain.T,(ytrain-self.p))
            self.weights += self.lam * (delta + linf)
            self.lam /= 5
            
                
    def predict(self, Xtest):
        #Xtest = Xtest2[:,:-1]
        xvec = np.dot(Xtest, self.weights)
        ytest = utils.sigmoid(xvec)
        ytest[ytest >= 0.5] = 1     
        ytest[ytest < 0.5] = 0
 	print("Logistic Q3:"),
        print time.time() - self.tlo

        return ytest
   
########################################################################                        
########################################################################                        

class NeuralNet(Classifier):
    """ Two-layer neural network; need to complete the inherited learn and predict functions """
    
    def __init__(self, params=None):
        # Number of input, hidden, and output nodes
        # Hard-coding sigmoid transfer for this implementation for simplicity
        self.ni = params['ni']
        self.nh = params['nh']
        self.no = params['no']
        self.transfer = utils.sigmoid
        self.dtransfer = utils.dsigmoid
        self.tneu = time.time()   

        # Set step-size
        self.stepsize = 0.01

        # Number of repetitions over the dataset
        self.reps = 5
        
        # Create random {0,1} weights to define features
        self.wi = np.random.randint(2, size=(self.nh, self.ni))
        self.wo = np.random.randint(2, size=(self.no, self.nh))

    def learn(self, Xtrain, ytrain):
        """ Incrementally update neural network using stochastic gradient descent """        
        for reps in range(self.reps):
            for samp in range(Xtrain.shape[0]):
                self.update(Xtrain[samp,:],ytrain[samp])

    def evaluate(self, inputs):
        """ Including this function to show how predictions are made """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = np.ones(self.nh)
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = np.ones(self.no)
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)


    def update(self, xi, yi):
        [ah, ao] = self.evaluate(xi)
        yhat = ao
        #print xi.shape #9,
        xi = np.reshape(xi.T,(1,xi.shape[0]))
        delta = (-(yi/yhat) + (1-yi)/(1-yhat)) * (yhat*(1-yhat))
        z = np.dot(self.wi, xi.T)  # (64*9) * (9*1)
        
        h = self.transfer(z)
        h2 = self.dtransfer(z)
        delta1 = delta * np.multiply(self.wo.T,h2)    
        #                   64x1       1x64  64x1
        delta2 = delta * h.T
        self.wi = self.wi - self.stepsize * delta1
        self.wo = self.wo - self.stepsize * delta2

                      
    def predict(self, Xtest):
	h = self.transfer(np.dot(self.wi, Xtest.T))
#add col of ones
        one = np.ones(h.shape[0])
        h[:,-1] = one
        #h[:,0] = one
#Adding column of ones at the start or end giving similar output
        yhat = self.transfer(np.dot(self.wo,h))

        yhat[yhat < 0.5] = 0
        yhat[yhat >= 0.5] = 1
        
	print("Neural Networks:"),
        print time.time() - self.tneu
        
        return yhat.T

