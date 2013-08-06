#!/usr/bin/python2.7
# -*- coding: UTF-8 -*-

#import numpy as np
#import tabular as tb
import sys
#import os
#import re
#import pandas as pd
#import scipy as sp
#import matplotlib as ml
#from pylab import plt
##from matplotlib import *
#from matplotlib.markers import MarkerStyle; mkStyles = MarkerStyle()

import mlUtils as mu

#fnameIn='data/indiegogo_4_viz_tmp.csv'
fnameIn='data/crowdfunding_4_viz_tmp.csv'
#table = pd.read_csv(fnameIn)
mlAnalysis = mu.tableModel(fnameIn)
#print 'table orig'; print table
#table = mlAnalysis.removeOutliers(table)
#print 'table post removeOutlier'; print table
# Full set ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountPledged', 'amountGoal', 'amountOver']

mlAnalysis.genScatter('amountPledged','amountGoal',  xrange=[-1000,11000], yrange=[-1000,11000])
mlAnalysis.genScatter('amountPledged','nbUpdates',   xrange=[-1000,11000], yrange=[-10,50])
mlAnalysis.genScatter('amountPledged','nbComments',  xrange=[-1000,11000], yrange=[-10,100])
mlAnalysis.genScatter('amountPledged','nbBackers',   xrange=[-1000,11000], yrange=[-10,1000])
mlAnalysis.genScatter('amountPledged','nbFriendsFB', xrange=[-1000,11000], yrange=[-10,1000])
fig, figax = mlAnalysis.genScatter('amountPledged','amountGoal')

mlAnalysis.genScatter('amountGoal','amountPledged',  xrange=[-1000,11000], yrange=[-1000,11000])
mlAnalysis.genScatter('nbUpdates','amountPledged',   xrange=[-10,50],   yrange=[-1000,11000])
mlAnalysis.genScatter('nbComments','amountPledged',  xrange=[-10,100],  yrange=[-1000,11000])
mlAnalysis.genScatter('nbBackers','amountPledged',   xrange=[-10,1000], yrange=[-1000,11000])
mlAnalysis.genScatter('nbFriendsFB','amountPledged', xrange=[-10,1000], yrange=[-1000,11000])

mlAnalysis.genPlot('amountGoal','amountPledged')
mlAnalysis.genPlot('nbUpdates','amountPledged')
mlAnalysis.genPlot('nbComments','amountPledged')
mlAnalysis.genPlot('nbBackers','amountPledged')
mlAnalysis.genPlot('nbFriendsFB','amountPledged')

mlAnalysis.genPlot('amountGoal','amountPledged',  xrange=[-1000,11000])
mlAnalysis.genPlot('nbUpdates','amountPledged',   xrange=[-10,50])
mlAnalysis.genPlot('nbComments','amountPledged',  xrange=[-10,100])
mlAnalysis.genPlot('nbBackers','amountPledged',   xrange=[-10,1000])
mlAnalysis.genPlot('nbFriendsFB','amountPledged', xrange=[-10,1000])

mlAnalysis.genHist('amountPledged')
mlAnalysis.genHist('amountGoal')
mlAnalysis.genHist('nbUpdates')
mlAnalysis.genHist('nbComments')
mlAnalysis.genHist('nbBackers')
mlAnalysis.genHist('nbFriendsFB')

mlAnalysis.genLinRegPlot('amountGoal','amountPledged',  xrange=[-1000,11000])
mlAnalysis.genLinRegPlot('nbUpdates','amountPledged',   xrange=[-10,50])
mlAnalysis.genLinRegPlot('nbComments','amountPledged',  xrange=[-10,100])
mlAnalysis.genLinRegPlot('nbBackers','amountPledged',   xrange=[-10,1000])
mlAnalysis.genLinRegPlot('nbFriendsFB','amountPledged', xrange=[-10,1000])



#features = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', lambda x: x**2] # 'amountOver'
features = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB'] # 'amountOver'
target = 'amountPledged'
kwargs={} # sizeTrainingSet=2500, n_iter=30
mlAnalysis.filterTrainVsTest(features=features, target=target,sizeTrainingSet=2500)
clf, tableProcDict = mlAnalysis.genModel( model='SGDRegressor', features=features, target=target, kwargs=kwargs)


#featuresCatAll = ['gender', 'supCat', 'location', 'country', 'platform']
#featuresNumAll = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal'] # 'amountOver'
features = ['nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB', 'amountGoal__amountGoal', 'nbFriendsFB__amountGoal', 'gender', 'supCat', 'country'] # 'amountOver'
#features = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', 'nbUpdates__nbUpdates', 'nbComments__nbComments', 'nbBackers__nbBackers', 'nbFriendsFB__nbFriendsFB', 'amountGoal__amountGoal'] # 'amountOver'
#features = ['nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB'] # 'amountOver'
target = 'amountPledged'

testCrit = [10, 30, 100, 300, 1000, 3000] # for sizeTrainingSet
testCrit = [10, 20, 30, 50, 100, 300, 600, 1000, 2000, 3000, 10000, 30000] # for sizeTrainingSet
#testCrit = [10, 20, 30, 50, 100, 300, 600] # for sizeTrainingSet
#testCrit = np.arange(5000,10000,300) # for sizeTrainingSet # 2 bumps to 1E14 with no shuffling.
#testCrit = np.arange(500,2000,100) # for sizeTrainingSet
#testCrit = np.arange(10,300,10) # for sizeTrainingSet
#testCrit = [10,300] # for sizeTrainingSet
model = 'Ridge' # 'LinearRegression', 'SGDRegressor','Ridge'
model = ['Ridge', 'LinearRegression' ] #, 'SGDRegressor','Ridge'
# alpha=0.0000000001, n_iter=100

#kwargs = {'alpha':0.001, 'n_iter':100, 'eta0':0.001, 'shuffle':False, 'removeOutliers':True} # eta0=0.01 default
kwargs = {'alpha':0.001, 'n_iter':100, 'eta0':0.001} # eta0=0.01 default
mlMultiTest = mu.tableModels(fnameIn)
mlMultiTest.baseModel.shuffle()
mlMultiTest.baseModel.removeOutliers('amountPledged', 50000)
targetNumCompare='amountGoal'
fig, figax = mlMultiTest.testModels(model, features, target, targetNumCompare, testCrit, kwargs)




if __name__ == "__main__":
    fnameIn='data/indiegogo_4_viz_tmp.csv' # may be overwritten below
    fnameIn='data/kickstarter_4_viz_tmp.csv' # may be overwritten below
    fnameIn='data/crowdcube_4_viz_tmp.csv' # may be overwritten below
    fnameIn='data/crowdfunding_4_viz_tmp.csv' # may be overwritten below
    if len(sys.argv) > 1: fnameIn=sys.argv[1]

    #table = tb.tabarray(SVfile=fnameIn)
    #table = pd.read_csv(fnameIn)

    # To run
    #python analyzeAll.py > data/indiegogo_4_viz_analysis.txt




"""
from sklearn import linear_model
clf = linear_model.LinearRegression()
X = [[0, 0], [1, 1], [2, 2]] # orig [[0, 0], [1, 1], [2, 2]]
y = [0, 1, 2]
clf.fit (X=X, y=y)
#LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
clf.coef_
#array([ 0.5,  0.5])

fig = plt.figure()
figax = fig.add_subplot(111)
figax.plot(X, y, c='r')
#figax.plot(x=X, y=clf.predict(X[:, np.newaxis]), c='b-') # X[:, np.newaxis] trick needed only if X is one dimensional as linear reg or predicte function requires X in 2D.
figax.plot(X, clf.predict(X), c='b')
plt.show()

from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
#SGDClassifier(alpha=0.0001, class_weight=None, epsilon=0.1, eta0=0.0, fit_intercept=True, l1_ratio=0.15, learning_rate='optimal', loss='hinge', n_iter=5, n_jobs=1, penalty='l2', power_t=0.5, random_state=None, rho=None, shuffle=False, verbose=0, warm_start=False)
clf.predict([[2., 2.]])

#x = table['nbUpdates']
#y = table['nbComments']
#scatter(x,y, marker='^', c='r')
#show()

#tablePD = convertTabarray2Panda(table)
#table = convertPanda2Tabarray(tablePD)

# Split from http://scikit-learn.github.io/scikit-learn-tutorial/general_concepts.html#classification
# Found a better way though: 
#indices = np.arange(len(X))
#indices[:10]
#array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#print 'Indices1: ',indices[:10]
#np.random.RandomState(42).shuffle(indices)
#indices[:10]
#array([ 73,  18, 118,  78,  76,  31,  64, 141,  68,  82])
#print 'Indices2: ',indices[:10]
#X = X[indices]
#y = y[indices]

"""
