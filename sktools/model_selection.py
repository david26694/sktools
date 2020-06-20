import numpy as np
from sklearn.utils import check_array


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

    References
    ----------

    .. [1] Out of sample data for bootstrap sample, from https://stats.stackexchange.com/questions/88980/

    """

    def __init__(self, n_bootstraps, size_fraction):

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
        sample_size = round(self.size_fraction * len(row_range), 0)

        for boot in range(self.n_bootstraps):
            train_idx = np.random.choice(row_range, sample_size)
            test_idx = list(set(row_range).difference(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):

        return self.n_bootstraps
