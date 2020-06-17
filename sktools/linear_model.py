from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm


class QuantileRegression:
    """Quantile regression wrapper

    It can work on sklearn pipelines

    Example
    -------
    >>> from sktools import QuantileRegression
    >>> from sklearn.datasets import load_boston
    >>> boston = load_boston()['data']
    >>> y = load_boston()['target']
    >>> qr = QuantileRegression(quantile=0.9)
    >>> qr.fit(boston, y)
    >>> qr.predict(boston)[0:5].round(2)
    array([34.87, 28.98, 34.86, 32.67, 32.52])

    """

    def __init__(self, quantile=0.5, add_intercept=True):

        self.quantile = quantile
        self.add_intercept = add_intercept
        self.regressor = None
        self.regressor_fit = None

    def preprocess(self, X):

        X = X.copy()
        if self.add_intercept:
            X = sm.add_constant(X)
        return X

    def fit(self, X, y):

        X = self.preprocess(X)

        self.regressor = QuantReg(y, X)
        self.regressor_fit = self.regressor.fit(q=self.quantile)

    def predict(self, X, y=None):

        X = self.preprocess(X)

        return self.regressor_fit.predict(X)
