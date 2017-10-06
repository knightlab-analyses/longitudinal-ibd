import pandas as pd
import numpy as np
import os

from scipy.stats import entropy, zscore
from skbio import DistanceMatrix
from skbio.diversity import beta_diversity, alpha_diversity


def load_mf(fn):
    _df = pd.read_csv(fn, sep='\t', dtype=str, keep_default_na=False,
                      na_values=[])
    _df.set_index('#SampleID', inplace=True)
    return _df


def gradient_distance(dm_, mf_, trajectory, gradient):
    """Get the distances over a gradient in a distance matrix

    Parameters
    ----------
    dm_ : skbio.DistanceMatrix
        Objects with the distances to calculate.
    mf_ : pd.DataFrame
        DataFrame object with the metadata described in
        `trajectory` and `gradient`.
    trajectory : str
        Name of the column in `mf_` that groups samples as trajectories.
    gradient : str
        Name of the column in `mf_` that srts the samples in the
        trajectories.

    Returns
    -------
    pd.Series
        Series with subsequent distances indexed by sample names,
        where each sample has the distance to the next sample over
        the gradient.

    Notes
    -----
    If there are repeated values in the gradient category for any given
    sample in the trajectory, this may lead to misleading results.
    """
    mf_ = mf_.copy()
    out = pd.Series(index=mf_.index)

    if not np.issubdtype(mf_[gradient].dtype, np.number):
        raise ValueError("The gradient category is not of a numeric dtype.")

    for subject in mf_[trajectory].unique():
        s_df = mf_[mf_[trajectory] == subject][gradient].copy()
        s_df.sort_values(inplace=True)

        samples = s_df.index.values

        for index in range(len(samples)-1):
            val = dm_[samples[index], samples[index+1]]
            out.set_value(samples[index], val)
    return out


# copied from blue-yonder/tsfresh 5c85a69cd681a551daff0fb747b732b822ca9a9a
def mean_autocorrelation(x):
    """
    Calculates the average autocorrelation (Compare to
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation), taken over
    different all possible lags (1 to length of x)

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: float
    """
    var = np.var(x)
    n = len(x)

    if abs(var) < 10**-10 or n == 1:
        return 0
    else:
        r = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
        r = r[0: (n - 1)] / np.arange(n - 1, 0, -1)
        return np.nanmean(r / var)


def variance_larger_than_standard_deviation(x):
    """
    Boolean variable denoting if the variance of x is greater than its standard
    deviation. Is equal to variance of x being larger than 1

    :param x: the time series to calculate the feature of
    :type x: pandas.Series
    :return: the value of this feature
    :return type: bool
    """
    return np.var(x) > np.std(x)
# END copied from blue-yonder/tsfresh


def rms(x):
    return np.sqrt(np.nanmean(np.square(x)))


def abs_mean_diff(x):
    return np.mean(np.abs(np.diff(x)))


def mean_diff__(x):
    return np.mean(np.diff(x))


def non_zero_samples(x):
    return (x > 0).sum()


# inspired from tsfresh
def absolute_sum_of_diff(x):
    return np.sum(np.abs(np.diff(x)))


def abs_energy(x):
    return np.sum(np.square(x))


class Sculptor(object):
    def __init__(self, biom_table, mapping_file, tree, gradient, trajectory,
                 name=None):

        # make this feature creation unique
        if name is None:
            self.name = ''.join([chr(i) for i in np.random.randint(65, 90, 9)])
        else:
            self.name = name

        if not set(biom_table.ids('sample')).issubset(mapping_file.index):
            raise ValueError("There are samples without metadata")

        # needed to make subsampling possible
        self._original_mf = mapping_file.loc[biom_table.ids('sample')].copy()
        self._original_bt = biom_table.copy()

        # holders of the subsampled data
        self.mapping_file = None
        self.biom_table = None

        # cache for alpha and beta diversity
        self._alpha_diversity_values = None
        self._beta_diversity_matrices = None

        self.tree = tree.copy()

        # make sure we can run tree-based metrics
        for n in self.tree.traverse():
            if n.length is None:
                n.length = 0

        if gradient not in self._original_mf.columns:
            raise ValueError("The gradient category %s is not part of the "
                             "sample metadata." % gradient)
        if trajectory not in self._original_mf.columns:
            raise ValueError("The trajectory category %s is not part of the "
                             "sample metadata." % trajectory)

        self.gradient = gradient
        self.trajectory = trajectory

        if not np.issubdtype(self._original_mf[self.gradient].dtype,
                             np.number):
            raise ValueError("The gradient category must be a numeric dtype")

        # alpha-diversity-level features
        self._alpha_metrics = ['faith_pd', 'shannon', 'observed_otus', 'osd',
                               'brillouin_d', 'berger_parker_d', 'dominance',
                               'strong', 'chao1']

        # beta-diversity-level features
        self._beta_metrics = ['canberra', 'braycurtis', 'jaccard',
                              'weighted_unifrac', 'unweighted_unifrac']

    def randomly_select(self, N):
        """Randomly select a list of N samples per trajectory

        Note, this method doesn't return any values, it however sets
        ``self.mapping_file`` and ``self.biom_table`` to the subsampled
        versions of the original dataset.

        Parameters
        ----------
        N : int
            The number of samples to select per trajectory.
        """

        samples = []

        for _, t_mf in self._original_mf.groupby(self.trajectory):

            # only subsample from trajectories with enough samples
            if len(t_mf) < N:
                continue

            # select the needed number of samples
            s = np.random.choice(t_mf.index, N, replace=False).tolist()
            samples.append(s)

        # flatten the list
        samples = sum(samples, [])

        # very important to keep these objects untouched
        self.mapping_file = self._original_mf.loc[samples].copy()
        self.biom_table = self._original_bt.filter(samples, axis='sample',
                                                   inplace=False)

    def _check_selection(self):
        if self.mapping_file is None or self.biom_table is None:
            raise ValueError("The dataset needs to be uniformly subsampled "
                             "before executing this function.")

    def compute_alpha_diversity(self):
        """Compute and cache alpha diversity values

        This data is computed for the full dataset, not for a specific
        subsampling. Therefore once it is computed, we can later subsample
        from these vectors directly.

        See Also
        --------
        Sculptor.compute_beta_diversity
        """
        # is what's returned from to_frame a new copy?
        features = self._original_mf[[self.trajectory, self.gradient]].copy()

        X = self._original_bt.matrix_data.toarray().astype(np.int).T

        for metric in self._alpha_metrics:
            if metric == 'faith_pd':
                kws = {'tree': self.tree,
                       'otu_ids': self._original_bt.ids('observation')}
            else:
                kws = {}

            features[metric] = alpha_diversity(metric, X,
                                               self._original_bt.ids('sample'),
                                               **kws)

        self._alpha_diversity_values = features

    def alpha_table(self, metrics=None, summary_stats=None):
        """Extract features from an alpha-diversity matrix

        Parameters
        ----------
        metrics : list, optional
            List of alpha diversity metric names or functions to calculate.  If
            ``None`` is passed, then ``self._alpha_metrics`` will be used.
        summary_stats : list, optional
            List of feature-extraction functions to process time-series. By
            default it uses: ``[absolute_sum_of_diff, abs_mean_diff,
            variance_larger_than_standard_deviation, abs_energy]``.

        Returns
        -------
        pd.DataFrame
            Matrix of features by subject (rows are subjects, columns are
            features based on the time-series of each alpha-diversity metric).
        """
        self._check_selection()

        if metrics is None:
            metrics = self._alpha_metrics

        if summary_stats is None:
            summary_stats = [absolute_sum_of_diff, abs_mean_diff,
                             variance_larger_than_standard_deviation,
                             abs_energy]

        if not set(metrics).issubset(set(self._alpha_metrics)):
            # by default self.compute_alpha_diversity() will compute the
            # metrics in self._alpha_metrics, if a metric the user wants is not
            # here, then it will need to be added to self._alpha_metrics before
            # it can be used
            raise ValueError("Cannot find one or more metrics, unsupported "
                             "action with the current caching mechanism.")

        if self._alpha_diversity_values is None:
            self.compute_alpha_diversity()

        # subset the precomputed table to only the metrics and metadata
        columns = metrics + [self.trajectory, self.gradient]
        features = self._alpha_diversity_values[columns]

        # subset the table to the samples we are using for this subsampling
        features = features.loc[self.biom_table.ids()].copy()
        features.sort_values(by=self.gradient, inplace=True)

        # compute the features from the tables themselves
        grouped = features.groupby([self.trajectory],
                                   as_index=False).aggregate(summary_stats)

        # merge the names and statistics
        grouped.columns = ['%s_%s' % i for i in grouped.columns]

        # remove the features that are indicated by the gradient category
        # as these are strictly not relevant to this analysis
        keep = [i for i in grouped.columns if not i.startswith(self.gradient)]
        grouped = grouped[keep]

        return grouped

    def compute_beta_diversity(self):
        """Compute and cache beta diversity values

        This method calculates a beta diversity distance matrix and saves it
        to a folder for re-use. The matrices are calculated based on the full
        dataset so that any subsample drawn from the full dataset can be
        fetched from these precomputed matrices.

        See Also
        --------
        Sculptor.compute_alpha_diversity
        """
        dir_fp = 'roc-curves/%s/cached-matrices/' % self.name
        os.makedirs(dir_fp, exist_ok=True)

        X = self._original_bt.matrix_data.toarray().astype(np.int).T

        self._beta_diversity_matrices = {}

        for metric in self._beta_metrics:
            fp = os.path.join(dir_fp, metric + '.full.txt')

            if os.path.exists(fp):
                distance_matrix = DistanceMatrix.read(fp)
            else:
                if metric in {'unweighted_unifrac', 'weighted_unifrac'}:
                    kws = {'tree': self.tree,
                           'otu_ids': self._original_bt.ids('observation')}
                else:
                    kws = {}

                distance_matrix = beta_diversity(metric, X,
                                                 self._original_bt.ids(),
                                                 **kws)

                distance_matrix.write(fp)

            self._beta_diversity_matrices[metric] = distance_matrix

    def beta_table(self, metrics=None, summary_stats=None):
        """Extract features from a beta-diversity matrix

        Parameters
        ----------
        metrics : list, optional
            List of beta diversity metric names or functions to calculate. If
            ``None`` is passed, then ``self._beta_metrics`` will be used.
        summary_stats : list, optional
            List of feature-extraction functions to process time-series. By
            default it uses: ``[np.mean, np.std]``.

        Returns
        -------
        pd.DataFrame
            Matrix of features by subject (rows are subjects, columns are
            features based on the time-series of each beta-diversity metric).
        """
        self._check_selection()

        if metrics is None:
            metrics = self._beta_metrics

        if self._beta_diversity_matrices is None:
            self.compute_beta_diversity()

        # This list was reduced to only these two features from a lot more
        # seems like most of the other features were too auto-correlated to be
        # useful for the classifier. Previously we had [abs_energy, np.sum,
        # np.median, np.std, rms, np.min, np.max, np.mean,
        # mean_autocorrelation]
        if summary_stats is None:
            summary_stats = [np.mean, np.std]

        if not set(metrics).issubset(self._beta_metrics):
            raise ValueError("Cannot find one or more metrics, unsupported "
                             "action with the current caching mechanism")

        # is what's returned from to_frame a new copy?
        features = self.mapping_file[self.trajectory].to_frame().copy()

        for metric in metrics:
            distance_matrix = self._beta_diversity_matrices[metric]
            features[metric] = gradient_distance(distance_matrix,
                                                 self.mapping_file,
                                                 self.trajectory,
                                                 self.gradient)

        grouped = features.groupby([self.trajectory],
                                   as_index=False).aggregate(summary_stats)

        # merge the names and statistics
        grouped.columns = ['%s_%s' % i for i in grouped.columns]

        # remove the features that are indicated by the gradient category
        # as these are strictly not relevant to this analysis
        keep = [i for i in grouped.columns if not i.startswith(self.gradient)]
        grouped = grouped[keep]

        return grouped

    def microbes_over_time(self, ids=None, summary_stats=None):
        """Create a feature table with significant microbe time-series

        Parameters
        ----------
        ids: list of str, optional
            List of feature identifiers that will be used to extract
            time-series features.
        summary_stats: list, optional
            List of feature-extraction functions to process time-series. By
            default it uses: ``[np.mean, abs_energy, non_zero_samples,
            abs_mean_diff]``.

        Returns
        -------
        pd.DataFrame
            Matrix of features by subject (rows are subjects, columns are
            features based on the time-series of each microbe in `ids`).
        """
        self._check_selection()

        if summary_stats is None:
            # summary_stats = [np.sum, np.mean, np.median, abs_energy,
            # mean_diff__]
            summary_stats = [np.mean, abs_energy, non_zero_samples,
                             abs_mean_diff]

        bt = self.biom_table.norm(inplace=False)

        if ids is not None:
            bt = bt.filter(ids, axis='observation',
                           inplace=False)

        # CLR approach
        # df = bt.to_dataframe().apply(lambda x: clr(x.to_dense() + 1)).
        # T.copy()
        df = bt.to_dataframe().T.to_dense().copy()

        df[self.trajectory] = self.mapping_file[self.trajectory]
        df[self.gradient] = self.mapping_file[self.gradient]

        df.sort_values(by=self.gradient, inplace=True)

        grouped = df.groupby(self.trajectory,
                             as_index=False).aggregate(summary_stats)

        # merge the names and statistics (i is a tuple of column names)
        grouped.columns = ['%s_%s' % i for i in grouped.columns]

        # remove the features that are indicated by the gradient category
        # as these are strictly not relevant to this analysis
        keep = [i for i in grouped.columns if not i.startswith(self.gradient)]
        grouped = grouped[keep]

        return grouped
