from itertools import product
from biom import Table, example_table as et

import numpy as np
import pandas as pd


def long_biom(biom_table, metadata=None, columns=None):
    """Create a tidy version of a BIOM table

    Parameters
    ----------
    biom_table : bt.Table
        BIOM Table to *elongate*.
    metadata : pd.DataFrame, optional
        Sample metadata about `biom_table` that should be
        added to the long version of the table.
    columns : list
        Information from `metadata` that should be added to
        the returning long table.

    Returns
    -------
    pd.DataFrame
        `biom_table` in tidy format, with additional columns
        as specified by `columns`, taken from `metadata`.

    Raises
    ------
    ValueError
        If the input `biom_table` does not have `taxonomy` metadata.
    """
    if 'taxonomy' not in biom_table.metadata(axis='observation')[0].keys():
        raise ValueError('Cannot operate on tables without taxonomy')

    index = ['SampleID', 'OTU_ID', 'Abundance', 'Kingdom', 'Phylum',
             'Class', 'Order', 'Family', 'Genus', 'Species']

    sample_df = []

    for o in biom_table.ids('observation'):
        data = ['', o, 0] + biom_table.metadata(o, 'observation')['taxonomy']
        sample_df.append(pd.Series(index=index, data=data))
    # concatenate the series and transpose to make the long format
    sample_df = pd.concat(sample_df, axis=1).T

    out = []
    for s in biom_table.ids('sample'):
        s_df = sample_df.copy()

        s_df['SampleID'] = s
        s_df['Abundance'] = biom_table.data(s, axis='sample')

        if columns:
            for col in columns:
                s_df[col] = metadata.loc[s][col]
        
        out.append(s_df)
        
    return pd.concat(out, axis=0)

