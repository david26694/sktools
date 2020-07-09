import numpy as np
from sklearn.utils import check_array
from sklearn.model_selection import GridSearchCV


class BootstrapFold:
    """Create folds based on bootsrapping

    For each fold, create a bootstrap sample, training data is the bootstrapped data.
    The test data is the rest of the data, the data that is not in the bootstrap sample

    The average size of the test data is 1/e of the total data.

    Parameters
    ----------

    n_bootstraps: int
        number of folds of our cross-validation setting
    size_fraction: float
        fraction of the training data being sampled. The lower, the bigger the test set
    Example
    -------
    >>> import numpy as np
    >>> from sktools.model_selection import BootstrapFold
    >>> X = np.array([
    >>>     np.random.randint(1, 3, 1000),
    >>>     np.random.randint(0, 2, 1000)]
    >>> ).T
    >>> loo = BootstrapFold(10, size_fraction=1)
    >>> for train_index, test_index in loo.split(X):
    >>>     print(f"Train length: {len(train_index)} Test length: {len(test_index)}")
    Train length: 1000 Test length: 393
    Train length: 1000 Test length: 367
    Train length: 1000 Test length: 372
    Train length: 1000 Test length: 377
    Train length: 1000 Test length: 361
    Train length: 1000 Test length: 356
    Train length: 1000 Test length: 366
    Train length: 1000 Test length: 369
    Train length: 1000 Test length: 390
    Train length: 1000 Test length: 365



    References
    ----------

    .. [1] Out of sample data for bootstrap sample, from https://stats.stackexchange.com/questions/88980/

    """

    def __init__(self, n_bootstraps=10, size_fraction=1):
        self.n_bootstraps = n_bootstraps
        self.size_fraction = size_fraction

    def split(self, X, y=None, groups=None):
        """
        Generator to iterate over the indices
        :param X: Array to split on
        :param y: Always ignored, exists for compatibility
        :param groups: Always ignored, exists for compatibility
        """

        X = check_array(X)

        row_range = range(X.shape[0])
        sample_size = int(round(self.size_fraction * len(row_range), 0))

        for boot in range(self.n_bootstraps):
            train_idx = np.random.choice(row_range, sample_size)
            test_idx = list(set(row_range).difference(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_bootstraps
