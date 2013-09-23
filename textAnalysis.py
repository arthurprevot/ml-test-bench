#!/usr/bin/python2.7

import numpy as np
import sys
#import os
#import re
import pandas as pd
#import scipy as sp
#import matplotlib as ml
#from pylab import plt
## Prep for nltk:
##os.system("export NLTK_DATA='/Volumes/Data/myDocs/A_Acad/TrainingPerso/hadoop/datasets/nltk-texts/'") # doesn't work
##os.environ['NLTK_DATA']='/Volumes/Data/myDocs/A_Acad/TrainingPerso/hadoop/datasets/nltk-texts/' # doesn't work
##So, has to run cmd before opening ipython: export NLTK_DATA='/Volumes/Data/myDocs/A_Acad/TrainingPerso/hadoop/datasets/nltk-texts/'
##print 'env:',os.environ['NLTK_DATA']
#import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
#import logging
#from optparse import OptionParser
#from time import time


class clustering:
    def __init__(self, fnameIn=None, fcontent=None):
        if type(fcontent) !=type(None): # assume it is panda table
            self.table = fcontent
        elif fnameIn:
            self.table = pd.read_csv(fnameIn)
        else:
            print 'pb with input to clustering class, fnameIn=%s, fcontent=%s'%(fnameIn, fcontent)
        self.cleanTable()

    def cleanTable(self):
        #self.table = self.table[0:5000] # DEBUG
        #self.table['descCampaign'].fillna('', inplace=True)
        self.table = self.table.dropna(subset=['descCampaign'], how='all')
        #return table

    def genVectorizer(self, listDocs):
        print "Extracting features from the training dataset using a sparse vectorizer (as X matrix will be sparsed.)"
        vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                                     stop_words='english')
        self.X = vectorizer.fit_transform(listDocs)
        print "n_samples: %d, n_features: %d" % self.X.shape

    def runKmeans(self, nClusters):
        #km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
        #                         init_size=1000,
        #                         batch_size=1000, verbose=1)

        km = KMeans(n_clusters=nClusters, init='random', max_iter=100, n_init=3,
                        verbose=1)
        print "Clustering sparse data with %s" % km
        km.fit(self.X)
        self.km = km

        # Add data
        print "self.table['clustNb']", len(self.table), len(self.km.labels_)
        self.table['clustNb'] = self.km.labels_ 

    def validate(self, colNameDocs, colNameLabels): # assumes we have set of categories defined and we check if clustering will get the same groups.
        table = self.table
        table = table[0:1000] # DEBUG
        
        catUniqs = np.unique(table[colNameLabels]) # outputs pd serie as input is pd series.
        catUniqs = catUniqs.tolist()
        table['clustLabels'] = table[colNameLabels].apply(lambda x: catUniqs.index(x))
        labels = np.asarray(table['clustLabels'])
        true_k = np.unique(labels).shape[0]

        self.table = table
        self.genVectorizer(table[colNameDocs]) #->self.X
        self.runKmeans(true_k)

        print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, self.km.labels_) # labels_true, and labels_pred
        print "Completeness: %0.3f" % metrics.completeness_score(labels, self.km.labels_)
        print "V-measure: %0.3f" % metrics.v_measure_score(labels, self.km.labels_)
        print "Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, self.km.labels_)
        print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(self.X, labels, sample_size=1000)

    def save(self, fnameOut):
        print 'Saved table to ',fnameOut
        self.table.to_csv(fnameOut)


if __name__ == "__main__":
    fnameIn = '/Volumes/Data/myDocs/A_Acad/TrainingPerso/Vagrant/vagrantGenPrecise64/scrapyCrowdfunding/outputData/output_agg/indiegogo_4_viz_tmp.csv' # may be overwritten below
    #fnameIn = '/Volumes/Data/myDocs/A_Acad/TrainingPerso/Vagrant/vagrantGenPrecise64/scrapyCrowdfunding/outputData/output_6_IndieWeb/indiegogo_4_viz_tmp_small3.csv' # may be overwritten below
    if len(sys.argv) > 1: fnameIn=sys.argv[1]
    fnameOut = '/Volumes/Data/myDocs/A_Acad/TrainingPerso/Vagrant/vagrantGenPrecise64/scrapyCrowdfunding/outputOther/indiegogo_4_viz_tmp_clustering.csv' # may be overwritten below
    #fnameOut = '/Volumes/Data/myDocs/A_Acad/TrainingPerso/Vagrant/vagrantGenPrecise64/scrapyCrowdfunding/outputData/output_6_IndieWeb/indiegogo_4_viz_tmp_small3_clustering.csv' # may be overwritten below
    if len(sys.argv) > 2: fnameIn=sys.argv[2]

    #table = pd.read_csv(fnameIn)
    myCluster = clustering(fnameIn)
    myCluster.validate(colNameDocs='descCampaign', colNameLabels='supCat')
    myCluster.save(fnameOut)

    # To run
    #python mlUtils.py data/crowdfunding_4_viz_tmp.csv > data/crowdfunding_4_viz_mlOut.txt
