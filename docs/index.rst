Welcome to sktools's documentation!
======================================

Tools to extend sklearn and make it more powerful. It mainly includes feature engineering tricks, but also helpers for pipelines.

Usage
-----

.. code-block:: python

  from sktools import IsEmptyExtractor

  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import Pipeline

  ...

  mod = Pipeline([
      ("impute-features", IsEmptyExtractor),
      ("model", LogisticRegression())
  ])

  ...


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   readme
   installation
   usage
   modules
   contributing
   authors
   history

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
