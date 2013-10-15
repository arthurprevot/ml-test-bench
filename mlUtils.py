#!/usr/bin/python2.7
# -*- coding: UTF-8 -*-

import numpy as np
import tabular as tb
import sys
import os
import re
import pandas as pd
import scipy as sp
import sklearn
from sklearn import linear_model
from sklearn import preprocessing
import matplotlib as ml
#from pylab import plt
import matplotlib.pyplot as plt
#from matplotlib import *
from matplotlib.markers import MarkerStyle; mkStyles = MarkerStyle()


class tableModel:
    def __init__(self, fnameIn=None, fcontent=None):
        if type(fcontent) !=type(None): # assume it is panda table
            self.dataOrig = fcontent
        elif fnameIn:
            self.dataOrig = pd.read_csv(fnameIn)
        #elif fnameInFeatures and fnameInTarget:
        #    pass #### TBD
        else:
            print 'pb with input to tableModel class, fnameIn=%s, fcontent=%s'%(fnameIn, fcontent)
        self.dataCur = self.dataOrig.copy()

    def removeOutliers(self, colName, threshold, useOrig=True):
        if useOrig:
            table = self.dataOrig.copy()
        else:
            table = self.dataCur.copy()
        table = table[table[colName] < threshold]
        self.dataCur = table
        return table

    def shuffle(self):
        table = self.dataCur.reindex(np.random.permutation(self.dataCur.index))
        return table

    def genScatter(self,xSource,ySource, colGroup=None, xrange=None, yrange=None, fnameOut='scatter.png'):
        table = self.dataCur
        fig, figax = genScatter(table,xSource,ySource,colGroup, xrange=xrange, yrange=yrange, fnameOut=fnameOut)
        return fig, figax

    def genHist(self,xSource, fnameOut='hist.png'):
        table = self.dataCur
        fig, figax = genHist(table,xSource=xSource,fnameOut=fnameOut)
        return fig, figax

    def genPlot(self,xSource, ySource, xrange=[]):
        table = self.dataCur
        fig, figax = genPlot(table,xSource, ySource, xrange=xrange)
        return fig, figax

    def genLinRegPlot(self,xSource, ySource, xrange=[]):
        table = self.dataCur
        fig, figax = genLinRegPlot(table,xSource, ySource, xrange=xrange)
        return fig, figax

    def computeCorrNbs(self, label1, label2):
        table = self.dataCur
        return computeCorrNbs(table,label1, label2)


    def clfPredAccuracy(self, targetCol, predTargetCol, runOn='all'):
        table = self.dataCur
        if runOn=='test':
            table=table[table['isTrainVsTest']==False]
        elif runOn=='train':
            table=table[table['isTrainVsTest']==True]

        totGood = sum(table[targetCol] == table[predTargetCol]) # may not account for NaN properly
        return float(totGood)/len(table)


    def filterTrainVsTest(self, features=[], target='n/a',splitRatio=0.66, sizeTrainingSet=None, shuffle=False):
        table = self.dataCur

        # Get features
        colsOrig = table.columns
        featuresExisting = [item for item in features if (type(item)==str and item in colsOrig)]
        featuresToGen    = [item for item in features if (type(item)!=str or item not in colsOrig)]
        print 'featuresExisting: ',featuresExisting, ', featuresToGen (if any): ',featuresToGen
        
        # ---------- Reduce table ----------
        table = table[featuresExisting+[target]]

        # ---------- Clean  ----------
        # Drop lines before removing extra cols
        #print 'len before nan drop', len(table)
        #table = table.fillna(0) # just to deal with NaN values for now but TODO: should remove corresponding rows instead.
        table = table.dropna(axis=0) # axis=0-> drop row. axis=1-> drop col
        #table = table[np.isnan(table['nbFriendsFB']) == False] # works too if we know all potential NaNs are only in 'nbFriendsFB'
        #print 'len after nan drop', len(table)

        if sizeTrainingSet!=None:
            table = table[:sizeTrainingSet] # assuming dropna above didn't remove too much.
            #table.to_csv('data/crowdfunding_5_ML_nbIterLast.csv') # debug

        # ---------- Gen new features  ----------
        # ---------- Gen features by multiplying other features ----------
        for item in featuresToGen:
            cols = item.split('__')
            table[item] = table[cols[0]]*table[cols[1]] # for now limited to multiplication of 2 cols. TODO: generalize.
        #print 'List features 1: ', table.columns; print table

        # ---------- Gen features from categorical cols ----------
        for item in featuresExisting[:]:
            #print 'Gen cat cols all: ', item, table.dtypes[item].name
            if table.dtypes[item].name == 'float64':
                continue

            # Gen one col per unique value in col
            #listCt = table[[item]].groupby([item]).first().to_records() # to_records to convert to numpy array.
            listCt = table[[item]].groupby([item]).first().index.tolist() # extract all unique values in column, to set as new columns.
            for item2 in listCt:
                #print 'item2: ', item2, item2[0]
                item2 = item2.replace(',','_')
                item2 = item2.replace(' ','_')
                item2 = item2.replace('"','_')
                item2 = item2.replace("'",'_')
                catColName = str(item2)+'_catGen'
                table[catColName] = np.where(table[item] == item2, 1, 0) # fill column with 1 or 0
                featuresExisting.append(catColName)
            # Then get rid of orig col
            table.__delitem__(item)
            featuresExisting.remove(item)
        #print 'List features 2: ', table.columns, ", note: 'otherIndex' and 'isTrainVsTest' added in filterTrainVsTest."

        # ---------- Shuffle (if not already done before) ----------
        if shuffle:
            # Note: to have same shuffle across test: need to put shuffle=False and do shuffle before serie of tests.
            table = table.reindex(np.random.permutation(table.index))

        # ---------- Mark Train vs Test sets ----------
        splitSp = len(table)*splitRatio
        table['otherIndex']=pd.Series(np.arange(len(table)), index=table.index) # to have clean index, sequential, as opposed to potentially shuffle one.
        table['isTrainVsTest']=table['otherIndex'] < splitSp # used by other functions later, and for later inspection.
        table.__delitem__('otherIndex')
        self.dataCur = table
        self.featureCols = featuresExisting
        self.targetCol = target
        #print 'List features 2: ', table.columns, ", note: 'isTrainVsTest' added in filterTrainVsTest." # and 'otherIndex'
        #return table

    def removeInvalidKwargs(self, model, kwargs):
        if model != 'SGDRegressor':
            if 'eta0' in kwargs.keys() : kwargs.pop('eta0')
            if 'n_iter' in kwargs.keys() : kwargs.pop('n_iter')

        if model == 'RandomForestRegressor':
            if 'alpha' in kwargs.keys() : kwargs.pop('alpha')
        return kwargs


    def genModel(self, model, features=None, target=None, kwargs={}):
        table = self.dataCur
        self.model = model

        # Get features and target col names (best to use these attached to class, i.e. set by filterTrainVsTest, as it is consistent with col used for normalizing.) TODO: check cases where good to override.
        if features == None: features = self.featureCols
        if target   == None: target   = self.targetCol

        table_train = table[table['isTrainVsTest'] == True]

        # Normalize & split sets  ----------
        X_train = table_train[features]
        y_train = table_train[target]
        #print 'X_train.columns',X_train.columns

        # Scale data:
        # SGDRegressor crashes without this scaling
        X_train_np = preprocessing.scale(X_train)
        X_train = pd.DataFrame(X_train_np, columns=X_train.columns)
        #print 'X_train.columns2',X_train.columns

        genTest = True
        if genTest:
            table_test  = table[table['isTrainVsTest'] == False]

            # Select features
            X_test = table_test[features]
            y_test = table_test[target]

            # Scale
            X_test_np = preprocessing.scale(X_test)
            X_test = pd.DataFrame(X_test_np, columns=X_test.columns)

        # Remove params from kwargs
        kwargs = self.removeInvalidKwargs(model, kwargs)

        # Run models
        from sklearn import linear_model
        from sklearn import ensemble
        from sklearn import naive_bayes
        if model == 'LinearRegression':
            clf = linear_model.LinearRegression(**kwargs) # normalize arg can't be used
        elif model == 'LogisticRegression':
            clf = linear_model.LogisticRegression(**kwargs) # normalize arg can't be used ?
        elif model == 'SGDRegressor':
            # input : alpha, n_iter, eta0
            clf = linear_model.SGDRegressor(**kwargs) # normalize arg can't be used.
        elif model == 'SGDClassifier':
            # input : alpha, n_iter, eta0
            clf = linear_model.SGDClassifier(**kwargs) # normalize arg can't be used. ?
        elif model == 'Ridge':
            # input : alpha
            clf = linear_model.Ridge(**kwargs) # can add normalize=True
        elif model == 'RidgeClassifier':
            # input : alpha
            clf = linear_model.RidgeClassifier(**kwargs) # can add normalize=True
        elif model == 'BayesianRidge':
            clf = linear_model.BayesianRidge(**kwargs) #
        elif model == 'Perceptron':
            clf = linear_model.Perceptron(**kwargs) #
        elif model == 'GaussianNB':
            clf = naive_bayes.GaussianNB(**kwargs) #
        elif model == 'MultinomialNB':
            clf = naive_bayes.MultinomialNB(**kwargs) #
        elif model == 'BernoulliNB':
            clf = naive_bayes.BernoulliNB(**kwargs) #
        elif model == 'DecisionTreeClassifier':
            clf = sklearn.tree.DecisionTreeClassifier(**kwargs) #
        elif model == 'DecisionTreeRegressor':
            clf = sklearn.tree.DecisionTreeRegressor(**kwargs) #
        elif model == 'RandomForestRegressor':
            clf = ensemble.RandomForestRegressor(**kwargs) # args ? n_jobs (for par proc)
        elif model == 'RandomForestClassifier':
            clf = ensemble.RandomForestClassifier(**kwargs) # args ? n_jobs (for par proc)
        elif model == 'GradientBoostingRegressor':
            clf = ensemble.GradientBoostingRegressor(**kwargs) # args ? n_jobs (for par proc)
        elif model == 'GradientBoostingClassifier':
            clf = ensemble.GradientBoostingClassifier(**kwargs) # args ? n_jobs (for par proc)
        else:
            print 'genModel: model not valid:',model

        clf.fit(X=X_train, y=y_train)
        #print 'coeff', clf.coef_

        tableProcDict = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}
        #TODO attach above var to self to not have to pass it, probably as self.dataProc
        #TODO: get kwargs out so it can be reported according to updated content.
        return clf, tableProcDict, kwargs
        #return clf

    def testModel(self, clf, tableProcDict, targetCol, targetNumCompare, testLabel):
        """Handle model for linear regression, or for linear regression leading to classification (i.e. numerical features) and for pure classification (based on discrete labels only)."""
        table = self.dataCur
        X_train=tableProcDict['X_train']
        X_test =tableProcDict['X_test']
        y_train=tableProcDict['y_train']
        y_test =tableProcDict['y_test']

        yp_train=clf.predict(X_train) # putting X_train.as_matrix() doesn't change output type (i.e. pandas dataframe)
        yp_test =clf.predict(X_test) # same

        if self.model == 'LogisticRegression':
            # Noticed predict function outputs NaN for 1, but need to confirm it is always the case. TODO: check real NaNs are not interpreted as 1.
            yp_train.fillna(value=1, inplace=True)
            yp_test.fillna(value=1, inplace=True)

        table[targetCol+'Pred']=pd.Series(np.hstack((yp_train,yp_test)), index=table.index) # for later inspection
        #table[targetCol+'Pred']=pd.Series(np.hstack((y_train,y_test)), index=table.index) # for debug
        
        if hasattr(clf,'coef_'): coeffs = clf.coef_
        else                   : coeffs = ['n/a']
        print 'test: %s, all table cols=%s'%(testLabel, sorted(table.columns))
        print 'test: %s, model=%s'%(testLabel, self.model)
        print 'test: %s, features cols=%s'%(testLabel, sorted(X_train.columns))
        print 'test: %s, features coefs=%s'%(testLabel, coeffs) # randomForest doesn't have it.

        if table[targetCol].dtype.name == 'float64' and targetNumCompare != None: # assuming table[targetCol+'Pred'] is also 'float64'
            #cost_train = np.sqrt(np.mean((yp_train-y_train)**2)) # same as rmse
            #cost_test  = np.sqrt(np.mean((yp_test-y_test)**2)) # same as rmse
            rmse_train = ml.mlab.rms_flat(yp_train-y_train)
            corr_train = sp.stats.pearsonr(yp_train, y_train)
            rmse_test = ml.mlab.rms_flat(yp_test-y_test)
            corr_test = sp.stats.pearsonr(yp_test, y_test)
            #print 'test: %s, cost_train=%s, cost_test=%s'%(testLabel, cost_train, cost_test) # commented as same as rmse
            print 'test: %s, rmse_train=%s, rmse_test=%s'%(testLabel, rmse_train, rmse_test)
            print 'test: %s, corr_train=%s, corr_test=%s'%(testLabel, corr_train, corr_test)

            table['madeIt']     = table[targetCol]        >= table[targetNumCompare]
            table['madeItPred'] = table[targetCol+'Pred'] >= table[targetNumCompare]

            clf_train = self.clfPredAccuracy('madeIt', 'madeItPred', runOn='train')
            clf_test  = self.clfPredAccuracy('madeIt', 'madeItPred', runOn='test')
            print 'test: %s, clf_train=%s, clf_test=%s'%(testLabel, clf_train, clf_test)
            results = {'rmse_train':rmse_train, 'rmse_test':rmse_test,
                       'corr_train':corr_train, 'corr_test':corr_test,
                       'clf_train':clf_train  , 'clf_test':clf_test,
                       'coeffs':coeffs}

        else:
            clf_train = self.clfPredAccuracy(targetCol, targetCol+'Pred', runOn='train')
            clf_test  = self.clfPredAccuracy(targetCol, targetCol+'Pred', runOn='test')
            print 'test: %s, clf_train=%s, clf_test=%s'%(testLabel, clf_train, clf_test)
            results = {'clf_train':clf_train  , 'clf_test':clf_test,
                       'coeffs':coeffs}

        #fnameOut='data/debugML_nbIter_%s.csv'%(testLabel) # DEBUG only
        #table.to_csv(fnameOut) # DEBUG only
        return table, results

    # testModels was put in other class. Should consider having methodes to deal with building a model based on multiple models (using bagging).
    # This might require having the class data being duplicated multiple time for each model (could require a lot of memory or to dump them as pickle).

class tableModels:
    """ Class for testing models under variour conditions (dataset sizes, solver, feature) and understand the impact of features (performance, convergence, overfitting, underfitting...).
    tableModel class might contain a model made of several models combined, so this one may not be the only one handling multiple models, but only one for testing only.
    """
    def __init__(self, fnameIn=None, fcontent=None):
        self.baseModel = tableModel(fnameIn=fnameIn,fcontent=fcontent)

    def testModels(self, model, featureCols, targetCol, targetNumCompare, sizeSet, kwargs):
        """Takes raw values or list of values to generate every combinations."""
        # Goal is to rebuild plot such as http://www.astroml.org/sklearn_tutorial/practical.html
        sizeSetOrig = sizeSet[:]

        # Generate all combinations
        if type(sizeSet)==int: sizeSet=[sizeSet] # to make it a list of one item
        if type(model)==str: model=[model] # same.
        if type(featureCols[0])==str: featureCols=[featureCols] # basic type expected is list of strings
        if type(kwargs)==dict: kwargs=[kwargs] # same.

        tests = [ [item] for item in sizeSet] # first one
        tests = [ item1+[item2] for item1 in tests for item2 in model]
        #tests = [ [item] for item in model]
        tests = [ item1+[item2] for item1 in tests for item2 in featureCols]
        tests = [ item1+[item2] for item1 in tests for item2 in kwargs] # sometimes duplicates setup for kwargs that will be discarded anyway. TODO: optimize.
    
        # Now iterate runs and fill table on params and results.
        columns=['model','featureCols','kwargs','sizeSet','rmse_train', 'rmse_test',
                 'corr_train', 'corr_test', 'clf_train', 'clf_test','coeffs']
        #dtype = np.dtype([('model','S20'),('featureCols','S20'),('kwargs','S20'),('sizeSet','i32'),('RMSE_train','f32'),('RMSE_test','f32')])
        #tableNp = np.empty((len(tests)*len(sizeSetOrig), len(dtype)), dtype=dtype) #
        #table=pd.DataFrame(tableNp, columns=columns) # columns=columns # didn't work, not sure why.
        table=pd.DataFrame({ 'model'       : 'n/a',
                             'featureCols' : 'n/a',
                             'kwargs'      : 'n/a',
                             'sizeSet'     : np.zeros((len(tests)),dtype='int32'),
                             'rmse_train'  : np.zeros((len(tests)),dtype='f32'),
                             'rmse_test'   : np.zeros((len(tests)),dtype='f32'),
                             'corr_train'  : np.zeros((len(tests)),dtype='f32'),
                             'corr_test'   : np.zeros((len(tests)),dtype='f32'),
                             'clf_train'  : np.zeros((len(tests)),dtype='f32'),
                             'clf_test'   : np.zeros((len(tests)),dtype='f32'),
                             'coeffs'      : 'n/a',
                            }, columns=columns) # , columns=columns to force order.
 
        ii = -1
        for (sizeSet, model, featureCols, kwargs) in tests:
        #for (model, featureCols, kwargs) in tests:
          #for sizeSet in sizeSetOrig:
            ii+=1
            print '###----- item %s in %s'%(sizeSet, sizeSetOrig)
            print '### items:',sizeSet,model,featureCols,kwargs
            modelCur = tableModel(fcontent=self.baseModel.dataCur)

            modelCur.filterTrainVsTest(featureCols, targetCol, sizeTrainingSet=sizeSet)
            clf, tableProcDict, kwargsUp = modelCur.genModel(model, kwargs=kwargs.copy())
            table2, results = modelCur.testModel(clf, tableProcDict, targetCol, targetNumCompare, sizeSet)
            #print results
            #print table.columns
            
            table['model'].iloc[ii]   = model
            table['featureCols'].iloc[ii] = featureCols
            table['kwargs'].iloc[ii]  = kwargsUp
            table['sizeSet'].iloc[ii] = sizeSet
            table['rmse_train'].iloc[ii] = results['rmse_train']
            table['rmse_test'].iloc[ii]  = results['rmse_test']
            table['corr_train'].iloc[ii] = results['corr_train'][0]
            table['corr_test'].iloc[ii]  = results['corr_test'][0]
            table['clf_train'].iloc[ii]  = results['clf_train']
            table['clf_test'].iloc[ii]   = results['clf_test']
            table['coeffs'].iloc[ii]     = str( zip(featureCols, results['coeffs']) )
        self.table = table

    def saveRunResults(self, fnameOut):
        table = self.table
        # Send table to file
        #fnameTableOut = 'tempo/table.csv'
        sendToFile = True
        if sendToFile and fnameOut != None:
            table.sort(['model'], inplace=True)
            table.to_csv(fnameOut) # , encoding='utf_32'

    def plotRunResults(self):
        table = self.table
        # Plot data, individual pair of curves RMSE_train and RMSE_test vs sizeSet for each other combination of params.
        fig = plt.figure()
        figax = fig.add_subplot(111)
        # Need to convert table content to str for groupby function below (i.e. so it is hashable)
        table['modelTmp'] = table['model'].apply(lambda x: str(x)) #.astype(str) 
        table['kwargsStr'] = table['kwargs'].apply(lambda x: str(x)) 
        table['featureColsStr'] = table['featureCols'].apply(lambda x: str(x))
        #table.to_csv('tempo/tableDEBUG.csv') ### DEBUG
        # Find uniq combination of all params except 'sizeSet'.
        tableUniq = table[['modelTmp','featureColsStr','kwargsStr']].groupby(['modelTmp','featureColsStr','kwargsStr']).first()
        #tableUniq.to_csv('tempo/tableDEBUG2.csv') ### DEBUG
        # Create a RMSE vs sizeSet plot (based on a tmp table) for each combination of unique other params.
        for ii in np.arange(len(tableUniq)):
            # Build table of every sizeSet datapoints for given other param combination.
            tableSm = table[(table['modelTmp']==tableUniq['modelTmp'].iloc[ii])
                          & (table['featureColsStr']==tableUniq['featureColsStr'].iloc[ii])
                          & (table['kwargsStr']==tableUniq['kwargsStr'].iloc[ii]) ]
            #TODO: Change to different shade of blue or green for each pass, to dissociate plots.
            figax.plot(tableSm['sizeSet'], tableSm['rmse_train'], 'g-')
            figax.plot(tableSm['sizeSet'], tableSm['rmse_test'], 'b-')
        figax.set_xlabel('size set')
        figax.set_ylabel('cost')
        figax.grid(True)
        plt.show()

        #return table



#-------------- Serie of util functions --------------------------
 
def genScatter(table,xSource,ySource, colGroup=None, xrange=None, yrange=None, fnameOut='scatter.png'):
    #statusRepr = {'success':{'color':'g'},
    #              'failure':{'color':'r'},
    #              #'other':{'color':'grey'}}
    colrList = ['g','r','b']
    #categsRepr = mkStyles.filled_markers
    if colGroup == None:
        categs = [1]
    else:
        categs = table.groupby([colGroup]).mean().index.tolist()

    fig = plt.figure()
    #figax = fig.add_subplot(121) # grid 1*2, and go in 1st box
    figax = fig.add_subplot(111) # grid 1*1, and go in 1st box
    figax.set_xlabel(xSource)
    figax.set_ylabel(ySource)
    for ii, item in enumerate(categs): # should be connected to statusRepr.keys()
        #TODO: change selection of color to assign green and red to success or failure or to 1 or 0, if two groups only.
        #colorMark = statusRepr[item]['color']
        colorMark = colrList[np.mod(ii,len(colrList))] # mod to loop through list of predef colors.
        if colGroup == None: # if section can be removed by used groupby groups and iterating through them.
            tbFilt = table
        else:
            tbFilt = table[table[colGroup]==item]
        xFilt = tbFilt[xSource]
        yFilt = tbFilt[ySource]
        figax.scatter(x=xFilt,y=yFilt, marker='o', c=colorMark)
    #figax.set_xscale('log')
    #figax.set_yscale('log')
    if xrange: figax.set_xlim(xrange)
    if yrange: figax.set_ylim(yrange)
#    figax2 = fig.add_subplot(122) # grid 1*2, and go in 2nd box
#    figax2.set_xlabel(xSource)
#    figax2.set_ylabel(ySource)
#    figax2.scatter(x=xSuccess,y=ySuccess, c='g')
#    figax2.scatter(x=xFailure,y=yFailure, c='r')
#    figax2.scatter(x=xOther,y=yOther, c='grey')
#    figax2.set_xlim([-1000,11000])
#    figax2.set_ylim(yrange)

    if fnameOut!=None: plt.savefig(fnameOut, format='png') # fnameOut="test.png"
    plt.show()
    return fig, figax

def genHist(table,xSource,fnameOut='hist.png'):
    fig = plt.figure()
    figax = fig.add_subplot(111)
    #xSource='amountPledged'
    test = [item for item in table[xSource] if not np.isnan(item)] # could use maskes.
    #n, bins, patches = figax.hist(x=test, bins=1000, facecolor='green') # table[xSource].tolist()
    n, bins, patches = figax.hist(x=test, bins=10000, facecolor='green', histtype='stepfilled', log=True) # log=True
    figax.set_xlabel(xSource)
    figax.set_ylabel('Nb Occurences')
    #figax.set_xscale('log')
    #figax.set_yscale('log')
    #figax.semilogy()
    #ax.set_title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    #ax.set_xlim(40, 160)
    #ax.set_ylim(0, 0.03)
    figax.grid(True)
    if fnameOut!=None: plt.savefig(fnameOut, format='png') # fnameOut="test.png"
    plt.show()
    return fig, figax

def genPlot(table,xSource, ySource, xrange=[]):
    fig = plt.figure()
    figax = fig.add_subplot(111)
    #xSource='amountPledged'
    #test = [item for item in table[xSource] if not np.isnan(item)] # could use maskes.
    #print table[xSource]
    #print table[ySource]
    figax.plot(x=table[xSource], y=table[ySource])
    figax.set_xlabel(xSource)
    figax.set_ylabel(ySource)
    #figax.set_xscale('log')
    #figax.set_yscale('log')
    #figax.semilogy()
    #figax.set_xlim(xrange)
    #figax.set_ylim([-1000,11000])
    figax.grid(True)
    plt.show()
    return fig, figax


def genLinRegPlot(table,xSource, ySource, xrange=[]):
    clf = linear_model.LinearRegression()
    #XCols = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal'] # 'amountOver'
    XCols = xSource
    yCol = ySource
    table2 = table
    #print 'len before nan drop', len(table2)
    table2 = table2.fillna(0) # just to deal with NaN values for now but TODO: should remove corresponding rows instead.
    #table2 = table2.dropna(axis=0) # axis=0-> drop row. axis=1-> drop col
    #print 'len after nan drop', len(table2)
    clf.fit (X=table2[XCols][:, np.newaxis], y=table2[yCol])
    x = table2[xSource]
    y=clf.predict(x[:, np.newaxis])

    fig = plt.figure()
    figax = fig.add_subplot(111)
    figax.plot(x, table2[ySource], 'go')
    figax.plot(x, y, 'b-')
    figax.set_xlabel(xSource)
    figax.set_ylabel(ySource)
    #figax.set_xscale('log')
    #figax.set_yscale('log')
    #figax.semilogy()
    #figax.set_xlim(xrange)
    #figax.set_ylim([-1000,11000])
    figax.grid(True)
    plt.show()
    return fig, figax

def outputTree(tree, fnameOut):
    from StringIO import StringIO
    out = StringIO()
    out = sklearn.tree.export_graphviz(tree, out_file=out) # to get one tree out of the 10 in randomForest: clf.estimators_[0].
    #print out.getvalue()
    fh = open(fnameOut,'w') # fnameOut = 'tempo/tree.dot'
    fh.write(out.getvalue())
    fh.close()

def computeCorrNbs(table, label1, label2):
    rmse_train = ml.mlab.rms_flat(table[label1]-table[label2])
    corr_train = sp.stats.pearsonr(table[label1], table[label2])
    results = {'rmse':rmse, 'corr':corr}
    #print results
    return results



if __name__ == "__main__":
    fnameIn='data/indiegogo_4_viz_tmp.csv' # may be overwritten below
    fnameIn='data/kickstarter_4_viz_tmp.csv' # may be overwritten below
    fnameIn='data/crowdcube_4_viz_tmp.csv' # may be overwritten below
    fnameIn='data/crowdfunding_4_viz_tmp.csv' # may be overwritten below
    if len(sys.argv) > 1: fnameIn=sys.argv[1]

    table = pd.read_csv(fnameIn)

    # To run
    #python mlUtils.py data/crowdfunding_4_viz_tmp.csv > data/crowdfunding_4_viz_mlOut.txt

    #table = removeOutliers(table)

    #genScatter('amountPledged','amountGoal',  xrange=[-1000,11000], yrange=[-1000,11000])
    #genScatter('amountPledged','nbUpdates',   xrange=[-1000,11000], yrange=[-10,50])
    #genScatter('amountPledged','nbComments',  xrange=[-1000,11000], yrange=[-10,100])
    #genScatter('amountPledged','nbBackers',   xrange=[-1000,11000], yrange=[-10,1000])
    #genScatter('amountPledged','nbFriendsFB', xrange=[-1000,11000], yrange=[-10,1000])
    #fig, figax = genScatter('amountPledged','amountGoal')

    #genScatter('amountGoal','amountPledged',  xrange=[-1000,11000], yrange=[-1000,11000])
    #genScatter('nbUpdates','amountPledged',   xrange=[-10,50],   yrange=[-1000,11000])
    #genScatter('nbComments','amountPledged',  xrange=[-10,100],  yrange=[-1000,11000])
    #genScatter('nbBackers','amountPledged',   xrange=[-10,1000], yrange=[-1000,11000])
    #genScatter('nbFriendsFB','amountPledged', xrange=[-10,1000], yrange=[-1000,11000])
    #
    #genPlot('amountGoal','amountPledged')
    #genPlot('nbUpdates','amountPledged')
    #genPlot('nbComments','amountPledged')
    #genPlot('nbBackers','amountPledged')
    #genPlot('nbFriendsFB','amountPledged')

    #genPlot('amountGoal','amountPledged',  xrange=[-1000,11000])
    #genPlot('nbUpdates','amountPledged',   xrange=[-10,50])
    #genPlot('nbComments','amountPledged',  xrange=[-10,100])
    #genPlot('nbBackers','amountPledged',   xrange=[-10,1000])
    #genPlot('nbFriendsFB','amountPledged', xrange=[-10,1000])

    #genHist('amountPledged')
    #genHist('amountGoal')
    #genHist('nbUpdates')
    #genHist('nbComments')
    #genHist('nbBackers')
    #genHist('nbFriendsFB')

    #genLinRegPlot('amountGoal','amountPledged',  xrange=[-1000,11000])
    #genLinRegPlot('nbUpdates','amountPledged',   xrange=[-10,50])
    #genLinRegPlot('nbComments','amountPledged',  xrange=[-10,100])
    #genLinRegPlot('nbBackers','amountPledged',   xrange=[-10,1000])
    #genLinRegPlot('nbFriendsFB','amountPledged', xrange=[-10,1000])

    #XCols = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', lambda x: x**2] # 'amountOver'
    #XCols = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB'] # 'amountOver'
    #yCol = 'amountPledged'
    #clf, table2, X_train, X_test, y_train, y_test = genModel(table, features=XCols, label=yCol, sizeTrainingSet=2500, n_iter=30)

    # Full set ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountPledged', 'amountGoal', 'amountOver']
    featuresCatAll = ['gender', 'supCat', 'location', 'country', 'platform']
    featuresNumAll = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal'] # 'amountOver'
    features = ['nbFriendsFB', 'amountGoal', 'nbFriendsFB__nbFriendsFB', 'amountGoal__amountGoal', 'nbFriendsFB__amountGoal', 'gender', 'supCat', 'country'] # 'amountOver'
    #XCols = ['nbUpdates', 'nbComments', 'nbBackers', 'nbFriendsFB', 'amountGoal', 'nbUpdates__nbUpdates', 'nbComments__nbComments', 'nbBackers__nbBackers', 'nbFriendsFB__nbFriendsFB', 'amountGoal__amountGoal'] # 'amountOver'
    target = 'amountPledged'

    testCrit = [10, 30, 100, 300, 1000, 3000] # for sizeTrainingSet
    testCrit = [10, 20, 30, 50, 100, 300, 600, 1000, 2000, 3000, 10000, 30000] # for sizeTrainingSet
    #testCrit = [10, 20, 30, 50, 100, 300, 600] # for sizeTrainingSet
    #testCrit = np.arange(5000,10000,300) # for sizeTrainingSet # 2 bumps to 1E14 with no shuffling.
    #testCrit = np.arange(500,2000,100) # for sizeTrainingSet
    #testCrit = np.arange(10,300,10) # for sizeTrainingSet
    #testCrit = [10,300] # for sizeTrainingSet
    model = 'Ridge' # 'LinearRegression', 'SGDRegressor','Ridge'
    # alpha=0.0000000001, n_iter=100

    kwargs = {'alpha':0.001, 'n_iter':100, 'eta0':0.001, 'shuffle':False, 'removeOutliers':True} # eta0=0.01 default
    table2, fig, figax = testModel(model, table, features, target, testCrit, kwargs)



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
