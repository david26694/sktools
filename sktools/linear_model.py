from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm


class QuantileRegression:
    def __init__(self, quantile=0.5, add_intercept=True):
        self.quantile = quantile
        self.add_intercept = add_intercept
        self.regressor = None
        self.regressor_fit = None

    def fit(self, X, y):

        X = self.preprocess(X)

        self.regressor = QuantReg(y, X)
        self.regressor_fit = self.regressor.fit(q=self.quantile)

    def predict(self, X, y=None):

        X = self.preprocess(X)

        return self.regressor_fit.predict(X)

    def preprocess(self, X):

        X = X.copy()
        if self.add_intercept:
            X = sm.add_constant(X)
        return X
