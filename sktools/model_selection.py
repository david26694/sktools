import numpy as np
from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.utils import check_array


class BootstrapKFold:
    """
    - Create folds based on provided cluster method
    :param cluster_method: Clustering method with fit_predict attribute
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
        print(row_range)

        for boot in range(self.n_bootstraps):
            train_idx = np.random.choice(row_range, sample_size)
            test_idx = list(set(row_range).difference(train_idx))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):

        return self.n_bootstraps


