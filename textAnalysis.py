#!/usr/bin/python2.7

#import numpy as np
#import tabular as tb
#import sys
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
#from sklearn.datasets import fetch_20newsgroups
#from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import Normalizer
#from sklearn import metrics
#from sklearn.cluster import KMeans, MiniBatchKMeans






class clustering:
    def __init__(self, fnameIn=None, fcontent=None):
        if type(fcontent) !=type(None): # assume it is panda table
            self.table = fcontent
        elif fnameIn:
            self.table = pd.read_csv(fnameIn)
        else:
            print 'pb with input to tableModel class, fnameIn=%s, fcontent=%s'%(fnameIn, fcontent)
        #self.dataCur = self.dataOrig.copy()

    def genVectors(self, colName, threshold, useOrig=True):
        if useOrig:
            table = self.dataOrig.copy()
        else:
            table = self.dataCur.copy()
        table = table[table[colName] < threshold]
        self.dataCur = table
        return table




### AP: started from user_guide-0.12-git.pdf (scikit manual)
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import logging
from optparse import OptionParser
import sys
from time import time
import numpy as np

## Display progress logs on stdout
#logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
#
## parse commandline arguments
#op = OptionParser()
#op.add_option("--no-minibatch",
#              action="store_false", dest="minibatch", default=True,
#              help="Use ordinary k-means algorithm.")
#
#print __doc__
#op.print_help()
#(opts, args) = op.parse_args()
#
#if len(args) > 0:
#    op.error("this script takes no arguments.")
#    sys.exit(1)
#
################################################################################
## Load some categories from the training set
#categories = [
#    'alt.atheism',
#    'talk.religion.misc',
#    'comp.graphics',
#    'sci.space',
#]
## Uncomment the following to do the analysis on all the categories
##categories = None
#print "Loading 20 newsgroups dataset for categories:"
#print categories
#dataset = fetch_20newsgroups(subset='all', categories=categories,
#                             shuffle=True, random_state=42)
#print "%d documents" % len(dataset.data)
#print "%d categories" % len(dataset.target_names)
#print
#
fnameIn = '/Volumes/Data/myDocs/A_Acad/TrainingPerso/Vagrant/vagrantGenPrecise64/scrapyCrowdfunding/outputData/output_agg/indiegogo_4_viz_tmp.csv'
#fnameIn = '/Volumes/Data/myDocs/A_Acad/TrainingPerso/Vagrant/vagrantGenPrecise64/scrapyCrowdfunding/outputData/output_6_IndieWeb/indiegogo_4_viz_tmp_small3.csv'
table = pd.read_csv(fnameIn)
table = table[0:1000]
table['descCampaign'].fillna('', inplace=True)

catUniqs = np.unique(table['supCat']) # outputs pd serie as input is pd series.
catUniqs = catUniqs.tolist()
table['clustLabels'] = table['supCat'].apply(lambda x: catUniqs.index(x))
labels = table['clustLabels']
true_k = np.unique(labels).shape[0]
#print "Extracting features from the training dataset using a sparse vectorizer"
#t0 = time()
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,
                             stop_words='english')
X = vectorizer.fit_transform(table['descCampaign'])
#print "done in %fs" % (time() - t0)
print "n_samples: %d, n_features: %d" % X.shape
print

###############################################################################
# Do the actual clustering
#km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
#                         init_size=1000,
#                         batch_size=1000, verbose=1)

km = KMeans(n_clusters=true_k, init='random', max_iter=100, n_init=1,
                verbose=1)
print "Clustering sparse data with %s" % km
#t0 = time()
km.fit(X)
#print "done in %0.3fs" % (time() - t0)
print
print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_)
print "Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_)
print "V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_)
print "Adjusted Rand-Index: %.3f" % \
        metrics.adjusted_rand_score(labels, km.labels_)
print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, sample_size=1000)
print



#if __name__ == "__main__":
#    fnameIn='data/indiegogo_4_viz_tmp.csv' # may be overwritten below
#    fnameIn='data/kickstarter_4_viz_tmp.csv' # may be overwritten below
#    fnameIn='data/crowdcube_4_viz_tmp.csv' # may be overwritten below
#    fnameIn='data/crowdfunding_4_viz_tmp.csv' # may be overwritten below
#    if len(sys.argv) > 1: fnameIn=sys.argv[1]
#
#    table = pd.read_csv(fnameIn)
#
#    # To run
#    #python mlUtils.py data/crowdfunding_4_viz_tmp.csv > data/crowdfunding_4_viz_mlOut.txt
