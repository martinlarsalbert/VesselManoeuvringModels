import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Split data in train test sets
    train test are runs in "ids"

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        labels
    test_ratio : float
        fraction of dataset taken as test dataset.
    seed : int
        seed must be specified as this can be run in a loop where the
        seed should be the same for all itterations.

    Returns
    -------
    [type]
        X_train, y_train, X_test, y_test, train_data
    """

    np.random.seed(seed)

    n_test = int(np.ceil(test_ratio * len(X)))
    index_test = np.random.permutation(X.index)[0:n_test]
    index_test = np.sort(index_test)

    X_test = X.loc[index_test]
    y_test = y.loc[index_test]
    X_train = X.drop(index=index_test)
    y_train = y.drop(index=index_test)

    train_data = X_train.copy()
    train_data["y"] = y_train

    test_data = X_test.copy()
    test_data["y"] = y_test

    return X_train, y_train, X_test, y_test, train_data


def train_test_split_run(X: pd.DataFrame, y: pd.Series, id: pd.Series, ids=[22774]):
    """Split data in train test sets
    train test are runs in "ids"

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        labels
    id : pd.Series
        series with run id connected to rows in X and y
    ids : list, optional
        Run ids for the test set, by default [22774]

    Returns
    -------
    [type]
        X_train, y_train, X_test, y_test, train_data
    """

    mask_test = id.isin(ids)

    X_test = X.loc[mask_test]
    y_test = y.loc[mask_test]
    X_train = X.loc[~mask_test]
    y_train = y.loc[~mask_test]

    train_data = X_train.copy()
    train_data["y"] = y_train

    test_data = X_test.copy()
    test_data["y"] = y_test

    return X_train, y_train, X_test, y_test, train_data


def train_test_split_exteme(
    X: pd.DataFrame,
    y: pd.DataFrame,
    data: pd.DataFrame,
    min_ratio=0.1,
    max_ratio=0.1,
    min_keys=["u"],
    max_keys=["v", "r", "delta"],
):
    """Split data in train test sets
    train set are the extreme values.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        labels
    id : pd.Series
        series with run id connected to rows in X and y
    ids : list, optional
        Run ids for the test set, by default [22774]

    Returns
    -------
    [type]
        X_train, y_train, X_test, y_test, train_data
    """

    mask_test = (
        data[max_keys].abs() >= (1 - max_ratio) * data[max_keys].abs().max()
    ).any(1) | (
        data[min_keys].abs() <= (1 + min_ratio) * data[min_keys].abs().min()
    ).any(
        1
    )

    X_test = X.loc[mask_test]
    y_test = y.loc[mask_test]
    X_train = X.loc[~mask_test]
    y_train = y.loc[~mask_test]

    train_data = X_train.copy()
    train_data["y"] = y_train

    test_data = X_test.copy()
    test_data["y"] = y_test

    return X_train, y_train, X_test, y_test, train_data


def train_predict(
    train_data: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    N_trainings=100,
    train_ratio=0.2,
) -> pd.DataFrame:
    """Train a model on subsets of "train_data" and then make a predictions on "X_test", "y_test"

    Parameters
    ----------
    train_data : pd.DataFrame

    X_test : pd.DataFrame
        test features
    y_test : pd.Series
        test label
    N_trainings : int, optional
        Number of models/sub sets on train dataset, by default 100
    train_ratio : float, optional
        Fraction of train data used on the subsets, by default 0.2

    Returns
    -------
    pd.DataFrame
        Data from predictions with all subsets
        Columns:
        y_hat : predicted value
        z : test value "correct/measured" value
        x : index of test data
        i : name of subset
        parameters : number of parameters used in the VMM
        (number of features)
    """

    N = len(train_data)
    n_train = int(np.ceil(train_ratio * N))
    predictions = []
    for i in range(N_trainings):

        sample = train_data.sample(n_train, replace=True)

        y_train = sample.pop("y")
        X_train = sample

        pred = pd.DataFrame()

        # Train
        model_ = sm.OLS(y_train, X_train)
        result = model_.fit()

        # Predict test
        y_hat = result.predict(X_test)

        pred["y_hat"] = y_hat
        pred["z"] = y_test.values
        pred["x"] = X_test.index
        pred["i"] = i
        pred["parameters"] = len(train_data.columns)
        predictions.append(pred)

    df_sample_predictions = pd.concat(predictions)
    return df_sample_predictions


def pivot_mean(df, key) -> pd.Series:
    """Make the pivot with "x" as index and subset name "i" as columns
    Then take the mean value of "key"

    Parameters
    ----------
    df : [type]
        [description]
    key : [type]
        [description]

    Returns
    -------
    pd.Series
        [description]
    """
    f_hat_x = df.pivot(index="x", columns="i", values=key).mean(axis=1)
    return f_hat_x


def variances(df):
    var = df.pivot(index="x", columns="i", values="y_hat").var(axis=1)
    return var


def expected(df_sample_predictions: pd.DataFrame, y_test: pd.Series):

    vmm_groups = df_sample_predictions.groupby(by="vmm", sort=False)
    f_hats = vmm_groups.apply(pivot_mean, key="y_hat").transpose()
    bias = f_hats.sub(y_test.values, axis=0)
    MSEs = vmm_groups.apply(pivot_mean, key="residual^2").transpose()
    df_variances = vmm_groups.apply(variances).transpose()

    return f_hats, bias, MSEs, df_variances


def errors(df_sample_predictions: pd.DataFrame, y_test: pd.Series):
    f_hats, bias, MSEs, df_variances = expected(
        df_sample_predictions=df_sample_predictions, y_test=y_test
    )

    df_errors = pd.DataFrame()
    df_errors["MSE"] = MSEs.mean()
    df_errors["bias^2"] = (bias ** 2).mean()
    df_errors["variance"] = df_variances.mean()

    return df_errors


def error_bars(df_errors: pd.DataFrame, smart_scale=True):
    fig, ax = plt.subplots()
    df_errors.plot.bar(
        y=["bias^2", "variance"], stacked=True, label=["$Bias^2$", "$Var$"], ax=ax
    )

    if smart_scale:
        ax.set_ylim(0, 3 * df_errors["MSE"].median())

    ax.set_title("MSE")
    return fig
