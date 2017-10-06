from unittest import TestCase, main
from skbio import DistanceMatrix, TreeNode
from io import StringIO
from biom import Table
from itertools import product

from extractor import gradient_distance, Sculptor

import pandas as pd
import numpy as np
import os
import shutil


class GradientDistancesTests(TestCase):
    def setUp(self):
        dm_data = [[0, 1, 2, 3, 5, 8, 13, 21],
                   [1, 0, 3, 5, 12, 44, 3, 4],
                   [2, 3, 0, 22, 3, 11, 1, 6],
                   [3, 5, 22, 0, 6, 33, 6, 7],
                   [5, 12, 3, 6, 0, 12, 3, 8],
                   [8, 44, 11, 33, 12, 0, 6, 6],
                   [13, 3, 1, 6, 3, 6, 0, 9],
                   [21, 4, 6, 7, 8, 6, 9, 0]]

        ids = ['subject_1_1', 'subject_2_1', 'subject_2_0', 'subject_3_1',
               'subject_3_4', 'subject_1_9', 'subject_1_3', 'subject_1_5']

        headers = ['LOL', 'Subject', 'Order']
        map_data = [['22', '1', '1'],
                    ['2', '2', '1'],
                    ['2', '2', '0'],
                    ['1', '3', '1'],
                    ['2', '3', '4'],
                    ['34', '1', '9'],
                    ['NA', '1', '3'],
                    ['11111', '1', '5']]

        self.dm = DistanceMatrix(dm_data, ids)
        self.mf = pd.DataFrame(map_data, ids, headers)

    def test_gradient_distances_exceptions(self):

        with self.assertRaises(ValueError):
            gradient_distance(self.dm, self.mf, 'Subject', 'Order')

    def test_gradient_distances(self):
        self.mf['Order'] = pd.to_numeric(self.mf['Order'])
        obs = gradient_distance(self.dm, self.mf, 'Subject', 'Order')
        exp = pd.Series([13, np.nan, 3, 6, np.nan, np.nan, 9, 6],
                        index=['subject_1_1', 'subject_2_1', 'subject_2_0',
                               'subject_3_1', 'subject_3_4', 'subject_1_9',
                               'subject_1_3', 'subject_1_5'])
        pd.util.testing.assert_series_equal(obs, exp)

class TestSculptor(TestCase):
    def setUp(self):

        # small synthetic dataset
        sample_ids = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                      's10', 's11']
        self.mf = pd.DataFrame(data=[
            ['fasting', '8', 'A'],
            ['fasting', '-1', 'A'],
            ['control', '1', 'B'],
            ['control', '2', 'B'],
            ['control', '3', 'B'],
            ['fasting', '2', 'A'],
            ['fasting', '11', 'A'],
            ['control', '4', 'B'],
            ['control', '5', 'B'],
            ['control', '90', 'B'],
            ['fasting', '19.9', 'A'],
            ], columns=['Treatment', 'Day', 'Host'], index=sample_ids)
        self.mf['Day'] = pd.to_numeric(self.mf['Day'], errors='coerce')

        otu_ids = [str(i) for i in range(1, 8)]
        data = np.array([[0.0, 2.0, 5.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 6.0, 9.0, 0.0, 4.0, 0.0],
            [2.0, 6.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0],
            [1.0, 0.0, 8.0, 9.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0],
            [0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 0.0],
            [9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        self.bt = Table(data.T, otu_ids, sample_ids)

        tree_string = ("((1:0.2, 2:0.1)3P:0.3, (((7:0.1, 8:0.1)7P:0.8, (5:0.2,"
                       " 6:0.2)8P:0.1)5P:0.1, (3:0.2, 4:0.7)6P:0.9)4P:0.3)"
                       "root;")
        self.tree = TreeNode.read(StringIO(tree_string))

        # assumes to be only directories
        self.to_delete = []

    def tearDown(self):
        for element in self.to_delete:
            shutil.rmtree(element, ignore_errors=True)

        # delete the directory only if it is empty
        try:
            os.rmdir('roc-curves')
        except (OSError, FileNotFoundError):
            pass

    def test_constructor(self):
        obs = Sculptor(self.bt, self.mf, self.tree, 'Day', 'Host', 'test-name')

        self.assertTrue(obs.mapping_file is None)
        self.assertTrue(obs.biom_table is None)

        self.assertEqual(obs.name, 'test-name')

        self.assertTrue(obs._alpha_diversity_values is None)
        self.assertTrue(obs._beta_diversity_matrices is None)

        pd.util.testing.assert_frame_equal(self.mf, obs._original_mf)

        np.testing.assert_equal(obs._original_bt.ids(), self.bt.ids())
        np.testing.assert_equal(obs._original_bt.ids('observation'),
                                self.bt.ids('observation'))

        a = [self.bt.data(i) for i in self.bt.ids()]
        b = [obs._original_bt.data(i) for i in obs._original_bt.ids()]

        np.testing.assert_allclose(a, b)

        # needed to allow for phylogenetic metrics
        for node in obs.tree.postorder():
            self.assertTrue(node.length is not None)

    def test_constructor_errors(self):
        with self.assertRaisesRegex(ValueError, 'The gradient category'):
            _ = Sculptor(self.bt, self.mf, self.tree, 'XXX', 'Host')

        with self.assertRaisesRegex(ValueError, 'The trajectory category'):
            _ = Sculptor(self.bt, self.mf, self.tree, 'Day', 'XXX')

        with self.assertRaisesRegex(ValueError, 'numeric dtype'):
            _ = Sculptor(self.bt, self.mf, self.tree, 'Treatment', 'Host')

        # create fake metadata
        self.bt.update_ids({i: i + 'xx' for i in self.bt.ids()}, inplace=True)
        with self.assertRaisesRegex(ValueError, 'without metadata'):
            _ = Sculptor(self.bt, self.mf, self.tree, 'Day', 'Host')

    def test_random_select(self):
        np.random.seed(0)
        obs = Sculptor(self.bt, self.mf, self.tree, 'Day', 'Host',
                       'random-select')

        self.assertTrue(obs.mapping_file is None)
        self.assertTrue(obs.biom_table is None)

        obs.randomly_select(3)

        # if we randomly select three samples there should be 6 in total
        self.assertTrue(len(obs.mapping_file) == 6)
        self.assertEqual(obs.biom_table.shape, (7, 6))

    def test_random_select_errors(self):
        obs = Sculptor(self.bt, self.mf, self.tree, 'Day', 'Host',
                       'random-select-errors')

        with self.assertRaisesRegex(ValueError, 'uniformly subsampled'):
            obs.alpha_table()

        with self.assertRaisesRegex(ValueError, 'uniformly subsampled'):
            obs.beta_table()

        with self.assertRaisesRegex(ValueError, 'uniformly subsampled'):
            obs.microbes_over_time()

    def test_alpha(self):
        skl = Sculptor(self.bt, self.mf, self.tree, 'Day', 'Host',
                       'test-alpha')
        np.random.seed(0)
        skl.randomly_select(5)

        obs = skl.alpha_table(['faith_pd', 'observed_otus'])

        self.assertTrue(skl._alpha_diversity_values is not None)

        columns = ['faith_pd_absolute_sum_of_diff', 'faith_pd_abs_mean_diff',
                   'faith_pd_variance_larger_than_standard_deviation',
                   'faith_pd_abs_energy', 'observed_otus_absolute_sum_of_diff',
                   'observed_otus_abs_mean_diff',
                   'observed_otus_variance_larger_than_standard_deviation',
                   'observed_otus_abs_energy']
        data = [[2.1999999999999993, 0.5499999999999998, 0.0,
                 23.919999999999995, 2, 0.5, False, 32],
                [2.200000000000001, 0.5500000000000003, 0.0, 6.760000000000001,
                 3, 0.75, False, 22]]

        exp = pd.DataFrame(data=data, index=pd.Index(['A', 'B'], name='Host'),
                           columns=columns)
        pd.util.testing.assert_frame_equal(obs, exp)

    def test_alpha_errors(self):
        skl = Sculptor(self.bt, self.mf, self.tree, 'Day', 'Host',
                       'random-select-errors')
        skl.randomly_select(5)
        with self.assertRaisesRegex(ValueError, 'find one or more metrics'):
            skl.alpha_table(metrics=['does_not_exist'])

    def test_beta(self):
        skl = Sculptor(self.bt, self.mf, self.tree, 'Day', 'Host',
                       'unittest-test-beta')
        path = 'roc-curves/%s/cached-matrices/' % skl.name

        # avoid any unwanted accidents
        self.to_delete.append('roc-curves/%s/' % skl.name)

        np.random.seed(0)
        skl.randomly_select(5)
        obs = skl.beta_table(['unweighted_unifrac', 'jaccard'])

        data = [[0.3927777777777778, 0.4126532637086283, 0.9375,
                 0.12499999999999999],
                [0.6557886557886559, 0.1365522219610505, 1.0, 0.0]]
        columns = ['unweighted_unifrac_mean', 'unweighted_unifrac_std',
                   'jaccard_mean', 'jaccard_std']
        exp = pd.DataFrame(data=data, columns=columns,
                           index=pd.Index(['A', 'B'], name='Host'))

        pd.util.testing.assert_frame_equal(obs, exp)

        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(os.path.join(path,
                                                    'unweighted_unifrac.full.'
                                                    'txt')))
        self.assertTrue(os.path.exists(os.path.join(path, 'jaccard.full.txt')))

    def test_beta_errors(self):
        skl = Sculptor(self.bt, self.mf, self.tree, 'Day', 'Host',
                       'unittest-beta-errors')
        self.to_delete.append('roc-curves/%s' % skl.name)
        skl.randomly_select(5)
        with self.assertRaisesRegex(ValueError, 'find one or more metrics'):
            skl.beta_table(metrics=['does_not_exist'])

    def test_microbes_over_time(self):
        skl = Sculptor(self.bt, self.mf, self.tree, 'Day', 'Host',
                       'microbes-over-time')
        np.random.seed(0)
        skl.randomly_select(5)

        obs = skl.microbes_over_time()

        metrics = ['mean', 'abs_energy', 'non_zero_samples', 'abs_mean_diff']
        columns = ['%s_%s' % (a, b) for a, b in product(range(1, 8), metrics)]
        index = ['A', 'B']

        self.assertEqual(obs.columns.tolist(), columns)
        self.assertEqual(obs.index.tolist(), index)
        self.assertEqual(obs.values.shape, (2, 28))


if __name__ == '__main__':
    main()
