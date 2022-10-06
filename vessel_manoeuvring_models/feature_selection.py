import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


def drop_multicollinearity(X: pd.DataFrame, limit_corr=0.9) -> pd.DataFrame:
    """Drop features with high correlation in a recursive manner

    Parameters
    ----------
    X : pd.DataFrame
        _description_
    limit_corr : float, optional
        correlation below this limit will not be dropped, by default 0.9

    Returns
    -------
    pd.DataFrame
        X with dropped columns
    """

    corr = X.corr().abs()
    corr_ = np.tril(corr, k=-1)
    corr_ = pd.DataFrame(corr_, index=X.columns, columns=X.columns)
    keys = corr_.max().sort_values(ascending=False).index

    keep_all = False

    while not keep_all:

        keep_all = True

        for key in keys:

            if not key in corr:
                continue

            buddy = corr_.loc[key].idxmax()
            if buddy != key:

                other_corr = corr[key].drop(index=[key, buddy])
                other_corr_budy = corr[buddy].drop(index=[key, buddy])

                if np.max([other_corr.max(), other_corr_budy.max()]) > limit_corr:

                    keep_all = False

                    if other_corr.max() > other_corr_budy.max():
                        drop = key
                    else:
                        drop = buddy

                    corr.drop(columns=[drop], inplace=True)
                    corr.drop(index=[drop], inplace=True)
                    corr_.drop(columns=[drop], inplace=True)
                    corr_.drop(index=[drop], inplace=True)

    return X[corr_.columns].copy()


def feature_imporance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Rank the feature importance in a linear regression
    Applying a min/max scaler and that using the absolute value parameter value to rank importance.

    Parameters
    ----------
    X : pd.DataFrame
        features
    y : pd.Series
        label

    Returns
    -------
    pd.Series
        Feature importance [0..1] in descending order.
    """

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_transform = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
    linear_regression = LinearRegression(fit_intercept=False)
    linear_regression.fit(X=X_transform, y=y)

    coeffs = pd.Series(linear_regression.coef_, index=X_transform.columns)
    importance = coeffs.abs().sort_values(ascending=False)
    importance = importance / importance.sum()
    return importance


class BestFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, min_importance=0.01):
        super().__init__()
        self.min_importance = min_importance

    def fit(self, X, y):

        importance = feature_imporance(X, y)

        mask = importance > self.min_importance

        self.features = list(importance[mask].index)

        if len(self.features) == 0:
            self.features = [importance.index[0]]

        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        return X[self.features].copy()


class DropCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, limit_corr=0.01):
        super().__init__()
        self.limit_corr = limit_corr

    def fit(self, X, y=None):

        X_ = drop_multicollinearity(X, limit_corr=self.limit_corr)
        self.features = list(X_.columns)
        return self

    def transform(self, X, y=None):
        # Perform arbitary transformation
        return X[self.features].copy()
