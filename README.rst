=======
sktools
=======


.. image:: https://img.shields.io/pypi/v/sktools.svg
        :target: https://pypi.python.org/pypi/sktools

.. image:: https://img.shields.io/travis/david26694/sktools.svg
        :target: https://travis-ci.com/david26694/sktools

.. image:: https://readthedocs.org/projects/sktools/badge/?version=latest
        :target: https://sktools.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

sktools provides tools to extend sklearn, like several feature engineering based transformers.

Installation
------------

To install sktools, run this command in your terminal:

.. code-block:: console

    $ pip install sktools


Documentation
-------------

Can be found in https://sktools.readthedocs.io


Usage
-----

.. code-block:: python

  from sktools import IsEmptyExtractor

  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import Pipeline

  ...

  mod = Pipeline([
      ("impute-features", IsEmptyExtractor()),
      ("model", LogisticRegression())
  ])

  ...



Features
--------

Here's a list of features that sktools currently offers:

* ``sktools.encoders.NestedTargetEncoder`` performs target encoding suited for variables with nesting.
* ``sktools.encoders.PercentileEncoder`` performs target aggregation using a quantile instead of the mean.
* ``sktools.impute.IsEmptyExtractor`` creates binary variables indicating if there are missing values.
* ``sktools.matrix_denser.MatrixDenser`` transformer that converts sparse matrices to dense.
* ``sktools.quantilegroups.GroupedQuantileTransformer`` creates quantiles of a feature by group.
* ``sktools.quantilegroups.PercentileGroupFeaturizer`` creates features regarding how an instance compares with a quantile of its group.
* ``sktools.quantilegroups.MeanGroupFeaturizer`` creates features regarding how an instance compares with the mean of its group.
* ``sktools.selectors.TypeSelector`` gets variables matching a type.
* ``sktools.selectors.ItemsSelector`` allows to manually choose some variables.
* ``sktools.ensemble.MedianForestRegressor`` applies the median instead of the mean when aggregating trees predictions.
* ``sktools.linear_model.QuantileRegression`` sklearn style wrapper for quantile regression.


License
-------

MIT license


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
