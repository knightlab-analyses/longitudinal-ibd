# Guiding longitudinal sampling in inflammatory bowel diseases cohorts

## [Supplementary Materials](Supplemental-Materials.ipynb)

This repository includes all the source code, tests and notebooks to generate
the figures used in the ([Vazquez-Baeza et al. 2017](http://gut.bmj.com/cgi/content/full/gutjnl-2017-315352)).

### Notes to the reader

While we do not provide all the data in this repository we provide the BIOM
tables, sample information, alpha and beta diversity tables. The raw and
quality controlled sequences from which the BIOM table originates can be found
in [Qiita study 2538](https://qiita.ucsd.edu/study/description/2538) (remember
to login).

**EBI Accession**: PRJEB23009 (ERP104742).

#### Notebooks

This is a brief description of computations contained in each notebook. The
first three sections depend on QIIME 1.9.1 and Python 2, the fourth section
depends on the scientific python stack and Python 3. Environment files used
to create these can be found in the `env-files` directory.

Note, for the QIIME 1.9.1 environment, you'll also need to install from source
the [diptest package](https://github.com/alimuldal/diptest), these notebooks
used the repository at SHA-1 bf005a8662d6e866842d5c0f387a011f773c5b04.

##### Setup

In the notebook [**01.1-setup.ipynb**](notebooks/01.1-setup.ipynb), we remove
blank samples, add alpha and beta diversity and add some of this information to
the metadata so it can be used in other notebooks. While we include the tabular
files with the collated alpha diversity, intermediate files and plots are not
included.

##### Alpha

In the notebook
[**02.1-alpha-diversity.ipynb**](notebooks/02.1-alpha-diversity.ipynb), we
calculate a few measures of alpha diversity, and compare distributions by
diagnosis of IBD and whether or not the subjects underwent surgery.

##### Beta

There are three steps to the use of beta diversity, first in
[**03.1-beta-diversity-stats.ipynb**](notebooks/03.1-beta-diversity-stats.ipynb)
we compare the groups using PERMANOVA and ANOSIM, then in
[**03.2-beta-diversity-distributions.ipynb**](notebooks/03.2-beta-diversity-distributions.ipynb)
we calculate the beta-diversity stability over time, and finally in
[**03.3-beta-diversity-regressions.ipynb**](notebooks/03.3-beta-diversity-regressions.ipynb)
we compare the microbial stability to the microbial dysbiosis index.

##### Classification

This section is the most computationally expensive. While prototyping, the
tests were executed through the Jupyter notebook interface, however to test the
pipeline with a reasonable number of iterations, we ran the comparisons
using a dedicated compute cluster using the [script provided
here](notebooks/featlib/pipeline.py).

In [**04.1-classification.ipynb**](notebooks/04.1-classification.ipynb) and
[**04.2-classification-jansson.ipynb**](notebooks/04.2-classification-jansson.ipynb),
we compare how good of a classification can you achieve depending on the number
of samples used per subject. The main difference between these two noebooks is
the data they use.

The ROC curves used in the paper are also included here, see
[notebooks/roc-curves/gg/](notebooks/roc-curves/gg/), while we expect to see
some variation from re-running this, we've observed that the same trends hold
(as reported in the paper).
