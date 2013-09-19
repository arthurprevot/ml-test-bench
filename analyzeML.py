#!/usr/bin/python2.7

import sys
import numpy as np
import mlUtils as mu
import sklearn

#fnameIn='data/indiegogo_4_viz_tmp.csv'
fnameIn='data/crowdfunding_4_viz_tmp.csv'
#fnameIn='/Volumes/Data/myDocs/A_Acad/TrainingPerso/Vagrant/vagrantGenPrecise64/scrapyCrowdfunding/outputData/output_agg/crowdfunding_4_viz_tmp.csv'
#table = pd.read_csv(fnameIn)


#----------- Gen plots to explore data ----------------------
print '#---- Gen plots to explore data ----'
mlAnalysis = mu.tableModel(fnameIn) # just loads data file.
#print 'table orig'; print table
#mlAnalysis.removeOutliers('amountPledged', 1000000)
#mlAnalysis.removeOutliers('amountGoal', 1000000)
if False: # to be switched manually
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


#----------- Run one ML algo ----------------------
print '#---- Run one ML algo ----'
# Full set ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountPledged', 'amountGoal', 'amountOver']
#features = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', lambda x: x**2] # 'amountOver'
features = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB'] # 'amountOver'
target = 'amountPledged'
kwargs={} # sizeTrainingSet=2500, n_iter=30
mlAnalysis.filterTrainVsTest(features=features, target=target,sizeTrainingSet=2500)
clf, tableProcDict = mlAnalysis.genModel( model='SGDRegressor', features=features, target=target, kwargs=kwargs)

# Test random forest.
mlAnalysis.filterTrainVsTest(features=features, target=target,sizeTrainingSet=2500)
clf2, tableProcDict2 = mlAnalysis.genModel( model='RandomForestRegressor', features=features, target=target, kwargs=kwargs)
from StringIO import StringIO
out = StringIO()
out = sklearn.tree.export_graphviz(clf2.estimators_[0], out_file=out) # to get one tree out of the 10.
#print out.getvalue()
fh = open('tempo/oneTreeFromForest.dot','w')
fh.write(out.getvalue())
fh.close()

#----------- Run multiple ML algos ----------------------
print '#---- Run multiple ML algo ----'
#featuresCatAll = ['gender', 'supCat', 'location', 'country', 'platform']
#featuresNumAll = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal'] # 'amountOver'
features = ['nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB', 'amountGoal__amountGoal', 'nbFriendsFB__amountGoal', 'gender', 'supCat', 'country'] # 'amountOver'
#features = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', 'nbUpdates__nbUpdates', 'nbComments__nbComments', 'nbBackers__nbBackers', 'nbFriendsFB__nbFriendsFB', 'amountGoal__amountGoal'] # 'amountOver'
#features = ['nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB'] # 'amountOver'

target = 'amountPledged'

#testCrit = [10, 30, 100, 300, 1000, 3000] # for sizeTrainingSet
#testCrit = [10, 20, 30, 50, 100, 300, 600, 1000, 2000, 3000, 10000, 30000] # for sizeTrainingSet
#testCrit = [10, 20, 30, 50, 100, 300, 600] # for sizeTrainingSet
#testCrit = np.arange(5000,10000,300) # for sizeTrainingSet # 2 bumps to 1E14 with no shuffling.
#testCrit = np.arange(500,2000,100) # for sizeTrainingSet
testCrit = np.arange(10,300,30) # for sizeTrainingSet
#testCrit = np.concatenate((np.arange(10,300,30),np.array([3000,6000, 15000, 80000]))) # for sizeTrainingSet
#testCrit = [10,300] # for sizeTrainingSet

#model = ['Ridge','LinearRegression','SGDRegressor','Ridge','RandomForestRegressor'] # Full
#model = ['Ridge', 'LinearRegression', 'SGDRegressor' ] #, 'SGDRegressor','Ridge'
model = ['Ridge', 'RandomForestRegressor' ] #, 'LinearRegression' is terrible, makes test data go to E17, probably because it is analytical solution and I have correlated features (rank 0) # 'SGDRegressor' goes wild after about sizeSet of 3K

# alpha=0.0000000001, n_iter=100
#kwargs = {'alpha':0.001, 'n_iter':100, 'eta0':0.001, 'shuffle':False, 'removeOutliers':True} # eta0=0.01 default
kwargs = {'alpha':0.001, 'n_iter':100, 'eta0':0.001} # eta0=0.01 default

mlMultiTest = mu.tableModels(fnameIn)
mlMultiTest.baseModel.shuffle()
mlMultiTest.baseModel.removeOutliers('amountPledged', 50000)
targetNumCompare='amountGoal'
table = mlMultiTest.testModels(model, features, target, targetNumCompare, testCrit, kwargs)
mlMultiTest.saveRunResults('tempo/table.csv')
mlMultiTest.plotRunResults()
