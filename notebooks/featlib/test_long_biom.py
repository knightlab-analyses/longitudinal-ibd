from unittest import TestCase, main

from long_biom import long_biom
from biom import Table, example_table as et

import pandas as pd
import numpy as np

class TopLevelTests(TestCase):
    def setUp(self):
        om = [{'taxonomy': ['foo', 'bar', 'baz', 'bazilus', 'foospiracea',
                            'spam', 'egg']},
              {'taxonomy': ['foo', 'bar', 'baz', 'bazilus', 'foospiracea',
                            'spam', 'gleep']},
              {'taxonomy': ['foo', 'bar', 'baz', 'bazilus', 'foospiracea',
                            'bloop', 'Boatface']}]
        self.test_table = Table(data=np.array([[1, 2], [3, 4], [5, 6]]),
                                observation_ids=['o1', 'o2', 'o3'],
                                sample_ids=['s1', 's2'],
                                observation_metadata=om)


        self.md = pd.DataFrame(data=[[True,  '++', '00001'],
                                     [False, ':L', '02341'],
                                     [True,  '8L', '00221'],
                                     [True,  '}8L', '00501']],
                               columns=['SampleID' 'IS_GOOD', 'LOL', 'PLEL'],
                               index=['s0', 's1', 's2', 's3'])

    def test_long_biom(self):
        long_t = long_biom(self.test_table)
        cols = ['SampleID', 'OTU_ID', 'Abundance', 'Kingdom', 'Phylum', 'Class',
                'Order', 'Family', 'Genus', 'Species']

        vals = np.array([['s1', 'o1', 1.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'spam', 'egg'],
                         ['s1', 'o2', 3.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'spam', 'gleep'],
                         ['s1', 'o3', 5.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'bloop', 'Boatface'],
                         ['s2', 'o1', 2.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'spam', 'egg'],
                         ['s2', 'o2', 4.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'spam', 'gleep'],
                         ['s2', 'o3', 6.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'bloop', 'Boatface']],
                        dtype=object)
        np.testing.assert_array_equal(long_t.values, vals)

        self.assertEqual(long_t.columns.tolist(), cols)

    def test_long_biom_edge(self):
        long_t = long_biom(self.test_table, self.md, ['LOL'])
        cols = ['SampleID', 'OTU_ID', 'Abundance', 'Kingdom', 'Phylum',
                'Class', 'Order', 'Family', 'Genus', 'Species', 'LOL']
        self.assertEqual(long_t.columns.tolist(), cols)

        vals = np.array([['s1', 'o1', 1.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'spam', 'egg', ':L'],
                         ['s1', 'o2', 3.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'spam', 'gleep', ':L'],
                         ['s1', 'o3', 5.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'bloop', 'Boatface', ':L'],
                         ['s2', 'o1', 2.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'spam', 'egg', '8L'],
                         ['s2', 'o2', 4.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'spam', 'gleep', '8L'],
                         ['s2', 'o3', 6.0, 'foo', 'bar', 'baz', 'bazilus',
                          'foospiracea', 'bloop', 'Boatface', '8L']],
                        dtype=object)
        np.testing.assert_array_equal(long_t.values, vals)

    def test_long_biom_exceptions(self):

        with self.assertRaises(ValueError):
            long_biom(et)

if __name__ == '__main__':
    main()
