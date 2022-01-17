from IPython.display import display
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


def show_pred(X, y, results, label):
    display(results.summary())
    plot_pred(X, y, results, label)


def plot_pred(X, y, results, label) -> plt.axes:

    X_ = X
    y_ = y
    y_pred = results.predict(X_)

    prstd, iv_l, iv_u = wls_prediction_std(results, exog=X_, alpha=0.05)
    # iv_l*=-1
    # iv_u*=-1

    fig, ax = plt.subplots()
    ax.plot(X_.index, y_, label="Numerical gradient from model test")
    ax.plot(X_.index, y_pred, "--", label="OLS")
    ax.set_ylabel(label)

    ax.fill_between(
        X_.index,
        y1=iv_l,
        y2=iv_u,
        zorder=-10,
        color="grey",
        alpha=0.5,
        label=r"5\% confidence",
    )
    ax.legend()
    return ax


def show_pred_captive(X, y, results, label):

    display(results.summary())
    plot_pred_captive(X, y, results, label)


def plot_pred_captive(X, y, results, label) -> plt.axes:

    X_ = X.copy()
    X_["y"] = y
    X_.sort_values(by="y", inplace=True)

    y_ = X_.pop("y")

    y_pred = results.predict(X_)

    prstd, iv_l, iv_u = wls_prediction_std(results, exog=X_, alpha=0.05)
    # iv_l*=-1
    # iv_u*=-1

    fig, ax = plt.subplots()
    # ax.plot(X_.index,y_, label='Numerical gradient from model test')
    # ax.plot(X_.index,y_pred, '--', label='OLS')

    ax.plot(y_, y_pred, ".")
    ax.plot([y_.min(), y_.max()], [y_.min(), y_.max()], "r-")

    ax.set_ylabel(f"{label} (prediction)")
    ax.set_xlabel(label)

    ax.fill_between(
        y_,
        y1=iv_l,
        y2=iv_u,
        zorder=-10,
        color="grey",
        alpha=0.5,
        label=r"5% confidence",
    )
    ax.legend()
    return ax
