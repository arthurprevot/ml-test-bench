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
mlAnalysis = mu.mlModel(fnameIn) # just loads data file.
#print 'table orig'; print table
#mlAnalysis.removeOutliers('amountPledged', 1000000)
#mlAnalysis.removeOutliers('amountGoal', 1000000)
if False: # to be switched manually
    mlAnalysis.genScatter('amountPledged','amountGoal','status',  xrange=[-1000,11000], yrange=[-1000,11000])
    mlAnalysis.genScatter('amountPledged','nbUpdates','status',   xrange=[-1000,11000], yrange=[-10,50])
    mlAnalysis.genScatter('amountPledged','nbComments','status',  xrange=[-1000,11000], yrange=[-10,100])
    mlAnalysis.genScatter('amountPledged','nbBackers','status',   xrange=[-1000,11000], yrange=[-10,1000])
    mlAnalysis.genScatter('amountPledged','nbFriendsFB','status', xrange=[-1000,11000], yrange=[-10,1000])
    #fig, figax = mlAnalysis.genScatter('amountPledged','amountGoal','status')

    mlAnalysis.genScatter('amountGoal','amountPledged','status',  xrange=[-1000,11000], yrange=[-1000,11000])
    mlAnalysis.genScatter('nbUpdates','amountPledged','status',   xrange=[-10,50],   yrange=[-1000,11000])
    mlAnalysis.genScatter('nbComments','amountPledged','status',  xrange=[-10,100],  yrange=[-1000,11000])
    mlAnalysis.genScatter('nbBackers','amountPledged','status',   xrange=[-10,1000], yrange=[-1000,11000])
    mlAnalysis.genScatter('nbFriendsFB','amountPledged','status', xrange=[-10,1000], yrange=[-1000,11000])

    fig, figax = mlAnalysis.genPlot('amountGoal','amountPledged') # not plottable because of no unique and incrementable values in x.
    mlAnalysis.genPlot('nbUpdates','amountPledged') # same
    mlAnalysis.genPlot('nbComments','amountPledged') # same
    mlAnalysis.genPlot('nbBackers','amountPledged') # same
    mlAnalysis.genPlot('nbFriendsFB','amountPledged') # same

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
    #sys.exit()


#----------- Run one ML algo ----------------------
print '#---- Run one ML algo ----'
# Full set ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountPledged', 'amountGoal', 'amountOver']
#features = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', lambda x: x**2] # 'amountOver'
features = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB'] # 'amountOver'
target = 'amountPledged'
kwargs={} # setSize=2500, n_iter=30
mlAnalysis.prepTable(features=features, target=target,setSize=2500)
skModel, tableProcDict, kwargUp = mlAnalysis.genModel( model='SGDRegressor', features=features, target=target, kwargs=kwargs)

# Test random forest.
mlAnalysis.prepTable(features=features, target=target,setSize=2500)
skModel2, tableProcDict2, kwargUp = mlAnalysis.genModel( model='RandomForestRegressor', features=features, target=target, kwargs=kwargs)
mu.outputTree(skModel2.estimators_[0], 'tempo/oneTreeFromForest.dot') # to get one tree out of the 10.


#----------- Run multiple ML algos ----------------------
print '#---- Run multiple ML algo ----'
#featuresCatAll = ['gender', 'supCat', 'location', 'country', 'platform']
#featuresNumAll = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal'] # 'amountOver'
features = ['nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB', 'amountGoal__amountGoal', 'nbFriendsFB__amountGoal', 'gender', 'supCat', 'country'] # 'amountOver'
#features = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', 'nbUpdates__nbUpdates', 'nbComments__nbComments', 'nbBackers__nbBackers', 'nbFriendsFB__nbFriendsFB', 'amountGoal__amountGoal'] # 'amountOver'
#features = ['nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB'] # 'amountOver'

target = 'amountPledged'

splitRatios = [0.01, 0.03, 0.1, 0.3]
#setSizes = [10, 30, 100, 300, 1000, 3000]
#setSizes = [10, 20, 30, 50, 100, 300, 600, 1000, 2000, 3000, 10000, 30000]
#setSizes = [10, 20, 30, 50, 100, 300, 600]
#setSizes = np.arange(5000,10000,300) # 2 bumps to 1E14 with no shuffling.
#setSizes = np.arange(500,2000,100)
#setSizes = np.arange(10,300,30)
setSizes = 10000
#setSizes = np.concatenate((np.arange(10,300,30),np.array([3000,6000, 15000, 80000])))
#setSizes = [10,300]

#model = ['Ridge','LinearRegression','SGDRegressor','Ridge','RandomForestRegressor'] # Full
#model = ['Ridge', 'LinearRegression', 'SGDRegressor' ] #, 'SGDRegressor','Ridge'
model = ['Ridge', 'RandomForestRegressor' ] #, 'LinearRegression' is terrible, makes test data go to E17, probably because it is analytical solution and I have correlated features (rank 0) # 'SGDRegressor' goes wild after about setSize of 3K

# alpha=0.0000000001, n_iter=100
#kwargs = {'alpha':0.001, 'n_iter':100, 'eta0':0.001, 'shuffle':False, 'removeOutliers':True} # eta0=0.01 default
kwargs = {'alpha':0.001, 'n_iter':100, 'eta0':0.001} # eta0=0.01 default
kwargs = [{'alpha':0.001, 'n_iter':100, 'eta0':0.001},{'alpha':0.002, 'n_iter':100, 'eta0':0.001}] # eta0=0.01 default

mlMultiTest = mu.mlModelsCompare(fnameIn)
mlMultiTest.baseModel.shuffle()
mlMultiTest.baseModel.removeOutliers('amountPledged', 50000)
targetNumCompare='amountGoal'
table = mlMultiTest.testModels(model, features, target, targetNumCompare, splitRatios, setSizes, kwargs)
mlMultiTest.saveRunResults('tempo/table.csv')
mlMultiTest.plotRunResults()
