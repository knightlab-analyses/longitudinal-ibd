#!/usr/bin/env python
"""This script is used to run the pipeline in a distributed computing system

Instead of running each test with a different number of samples per subject
one at a time, this script allows for parallel submission of multiple jobs each
with a different number of samples.

The notebooks and featlib are intended to provide the needed
context/information that may have been omitted in this file.
"""

import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import qiime2 as q2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

from scipy import interp

from skbio import TreeNode
from biom import load_table, Table
from featlib import Sculptor, load_mf


@click.command()
@click.option('--otu-fp', help='A rarefied OTU table (a BIOM file).')
@click.option('--metadata-fp', help='Metadata mapping file (a tab-delimitted '
              'file).')
@click.option('--tree-fp', help='Phylogenetic tree (a QIIME 2 artifact).')
@click.option('--samples-per-subject', help='Number of samples per subject',
              type=click.IntRange(0, 11))
@click.option('--method', type=click.Choice(['gg', 'gg-smaller', 'sepp']),
              help='Data to use gg (Greengenes 97% OTUs and clade names), '
              'gg-smaller (Greengenes 97% OTUs limited to the OTU IDs found '
              'by phylofactor), and sepp (deblurred OTUs using clade names).')
def main(otu_fp, metadata_fp, tree_fp, samples_per_subject, method):

    # rarefy before renaming
    bt = load_table(otu_fp)

    if method == 'sepp':
        bt.update_ids({i:i.replace('.fastq', '').split('.')[0]
                       for i in bt.ids('sample')}, 'sample', inplace=True)

    mf = load_mf(metadata_fp)

    # we only keep the samples that have sequences in the table
    mf = mf.loc[bt.ids()].copy()

    tree = q2.Artifact.load(tree_fp).view(TreeNode)
    for n in tree.traverse():
        if n.length is None:
            n.length = 0

    mf['DAYS_SINCE_EPOCH'] = pd.to_numeric(mf['DAYS_SINCE_EPOCH'],
                                           errors='coerce')

    if method == 'gg-smaller':
        # this file was provided by Alex
        otus = pd.read_csv('Proteos_Lachnos_Phylofactored_in_Terminal_Ileum.csv')
        pts_lchns = {str(int(i)) for i in set(otus.Proteos) | set(otus.Lachnos) if not np.isnan(i)}

    think = Sculptor(biom_table=bt, mapping_file=mf, tree=tree,
                     gradient='DAYS_SINCE_EPOCH', trajectory='HOST_SUBJECT_ID',
                     name=method)


    lw = 2

    N = 100
    N_samples = samples_per_subject
    plt.figure()
    i = 0
    while i < N:
        # taken from:
        # http://scikit-learn.org/stable/auto_examples/model_selection/
        # plot_roc_crossval.html
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)

        ## BEGIN feature creation
        think.randomly_select(N_samples)

        alpha = think.alpha_table(['faith_pd', 'chao1', 'brillouin_d'])
        beta = think.beta_table()

        features_to_keep = []

        if method in {'gg', 'sepp'}:
            for _, id_, md in think.biom_table.iter(axis='observation',
                                                    dense=False):
                t = md['taxonomy']
                if (t[4].lower() == 'f__lachnospiraceae'
                    or t[2].lower() == 'c__gammaproteobacteria'
                    or t[2].lower() == 'c__betaproteobacteria'):
                    features_to_keep.append(id_)
        else:
            features_to_keep = set(think.biom_table.ids('observation')) & pts_lchns

        # more than one sample
        if N_samples > 1:
            alpha = think.alpha_table(['faith_pd', 'chao1', 'brillouin_d'])
            beta = think.beta_table()
            features = think.microbes_over_time(ids=features_to_keep)

            # combine the data
            combined_features = pd.concat([features, alpha, beta], axis=1)

            # get a column with IBD status for all the subjects
            combined_features.dropna(axis=1, how='any', inplace=True)
            classes = think.mapping_file.groupby(['HOST_SUBJECT_ID', 'IBD'],
                                        as_index=False).aggregate(np.sum).set_index('HOST_SUBJECT_ID',
                                                                                    inplace=False)
            combined_features['IBD'] = classes['IBD']
        # one sample with our model
        elif N_samples == 1:
            alpha = think.alpha_table(['faith_pd', 'chao1', 'brillouin_d'], [abs_energy])
            features = think.biom_table.filter(ids_to_keep=features_to_keep, axis='observation')
            features = features.norm(inplace=False).to_dataframe().to_dense().T
            features['HOST_SUBJECT_ID'] = think.mapping_file['HOST_SUBJECT_ID']
            features['IBD'] = think.mapping_file['IBD']
            features.set_index('HOST_SUBJECT_ID', inplace=True)
            combined_features = pd.concat([features, alpha], axis=1)
        # one sample with only relative abundances
        elif N_samples == 0:
            combined_features = think.biom_table.norm(inplace=False).to_dataframe().to_dense().T
            combined_features['IBD'] = think.mapping_file['IBD']

        # get a list of the columns that will be used as features
        no_ibd = combined_features.columns.tolist()
        # no_ibd = smaller.columns.tolist()
        no_ibd.remove('IBD')
        ## END feature creation

        X_train, X_test, Y_train, Y_test = train_test_split(combined_features[no_ibd],
                                                            combined_features['IBD'],
                                                            test_size=0.35)

        clf = RandomForestClassifier(n_estimators=500)

        probas_ = clf.fit(X_train, Y_train).predict_proba(X_test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1],
                                         pos_label='Healthy Controls')

        # nans ruin the ROC curve, this happens when the train/test split
        # only includes one class in the training set
        if np.any(np.isnan(fpr)) or np.any(np.isnan(tpr)):
            continue
        else:
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            plt.plot(fpr, tpr, lw=0.1*lw, color='lightgray')

            i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= N
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc-curves/%s/%d.samples.per.subject-%d.iterations.pdf' % (think.name, N_samples, N))

if __name__ == '__main__':
    main()
