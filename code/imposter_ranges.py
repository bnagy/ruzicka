"""
For each dataset, this script runs a series of experiments
on the train corpus, using the Imposters Framework.
For all metrics and vector space combinations,
we collect PAN scores for a range of MFI (auc, c@1, etc.).
For each experiment, we score the score shifter's
p1 and p2, which yielded the optimal AUC x c@1.
"""

from __future__ import print_function
import os
import json
import pickle

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

from ruzicka.experimentation import dev_experiment
from ruzicka.utilities import get_vocab_size


# set hyperparameters:
corpus_dirs = ['../data/2014/du_essays/',
               '../data/2014/gr_articles/',
               '../data/2014/sp_articles/',
               '../data/2014/du_reviews/',
               '../data/2014/en_essays/',
               '../data/2014/en_novels/',
              ]
nb_experiments = 20
ngram_type = 'word'
ngram_size = 1
base = 'instance'
nb_bootstrap_iter = 100
rnd_prop = 0.5
min_df = 1
nb_imposters = 30

# create a dict, where we store the
# optimal settings and results
#for each metric and space pair:
best_settings = {}

for corpus_dir in corpus_dirs:
    print('>>> corpus:', corpus_dir)
    best_settings[corpus_dir] = {}
    
    # get max nb of features:
    max_vocab_size = get_vocab_size(corpus_dir = corpus_dir,
                                    ngram_type = ngram_type,
                                    ngram_size = ngram_size,
                                    min_df = min_df,
                                    phase = 'train')
    print('\t > vocab size:', max_vocab_size)
    
    for vector_space in ('tf_std', 'tf_idf', 'tf'):
        print('\t +++', vector_space)
        best_settings[corpus_dir][vector_space] = {}

        # new plot:
        sb.plt.clf()
        f, ax = plt.subplots(1,1)
        sb.set_style("darkgrid")
        ax.set_ylim(.0, 1)

        for metric in ('manhattan', 'minmax', 'euclidean'):
            print('\t\t>>>', metric)
            scores, p1s, p2s = [], [], []
            feature_ranges = [int(i) for i in np.linspace(100, max_vocab_size, nb_experiments)]
            print(feature_ranges)

            for nb_feats in feature_ranges:
                print('\t\t\t- nb feats:', nb_feats)
                dev_auc_score, dev_acc_score, dev_c_at_1_score, opt_p1, opt_p2 = \
                                    dev_experiment(corpus_dir = corpus_dir+'/',
                                                   mfi = nb_feats,
                                                   vector_space = vector_space,
                                                   ngram_type = ngram_type,
                                                   ngram_size = ngram_size,
                                                   metric = metric,
                                                   base = base,
                                                   min_df = min_df,
                                                   nb_bootstrap_iter = nb_bootstrap_iter,
                                                   rnd_prop = rnd_prop,
                                                   nb_imposters = nb_imposters)
                scores.append(dev_auc_score * dev_c_at_1_score)
                p1s.append(opt_p1)
                p2s.append(opt_p2)

            # determine position of optimal AUC x c@1:
            opt_idx = np.argmax(scores)
            # store best settings:
            opt_mfi = feature_ranges[opt_idx]
            opt_score = scores[opt_idx]
            opt_p1 = p1s[opt_idx]
            opt_p2 = p2s[opt_idx]
            best = {'score':opt_score, 'mfi':opt_mfi, 'p1':opt_p1, 'p2':opt_p2}
            best_settings[corpus_dir][vector_space][metric] = best
            opt_score = format(opt_score*100, '.2f')
            l = metric+' ('+str(opt_score)+' @ '+str(opt_mfi)+')'

            # plot results:
            sb.plt.plot(feature_ranges, scores, label=l)

        sb.plt.title(vector_space.replace('_', '-'))
        sb.plt.xlabel('# MFI')
        sb.plt.ylabel('AUC $\cdot$ c@1')
        sb.plt.legend(loc='best')
        c = os.path.basename(corpus_dir[:-1])
        sb.plt.savefig('../output/'+c+'_'+vector_space+'.pdf')

# dump best settings for reuse during testing:
with open('../output/best_train_params.json', 'w') as fp:
    json.dump(best_settings, fp)