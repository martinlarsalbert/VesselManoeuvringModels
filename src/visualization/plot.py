import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


standard_styles = ["b", "r", "g", "m", "c"]


def plot(
    dataframes: dict,
    subplot=True,
    fig_size=(10, 10),
    styles: list = None,
    keys: list = None,
):

    if keys is None:
        keys = set()
        for label, df in dataframes.items():
            keys = keys | set(df.columns)

    if subplot:
        number_of_axes = len(keys)
        ncols = 2
        nrows = int(np.ceil(number_of_axes / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
        fig.set_size_inches(fig_size)
        axes = axes.flatten()

    plot_kwargs = {}

    if isinstance(styles, list):
        plot_kwargs = {
            label: {"style": style} for label, style in zip(dataframes.keys(), styles)
        }
    elif isinstance(styles, dict):
        plot_kwargs = styles

    standard_styles_ = standard_styles.copy()
    for key in dataframes.keys():
        if not key in plot_kwargs:
            plot_kwargs[key] = {}

        if not "style" in plot_kwargs[key]:
            if len(standard_styles) > 0:
                standard_style = standard_styles_.pop(0)
            else:
                standard_style = standard_styles_[0]

            plot_kwargs[key]["style"] = standard_style

    for i, key in enumerate(sorted(keys)):
        if subplot:
            ax = axes[i]
        else:
            fig, ax = plt.subplots()

        for label, df in dataframes.items():

            plot_kwarg = plot_kwargs.get(label, {})

            if key in df:
                df.plot(y=key, label=label, ax=ax, **plot_kwarg)

        legend = ax.get_legend()
        if legend:
            legend.set_visible(False)
        ax.set_ylabel(key)

    lines = [len(ax.lines) for ax in axes]
    i = np.argmax(lines)
    axes[i].legend()

    plt.tight_layout()
    return fig


def track_plots(
    dataframes: dict,
    lpp: float,
    beam: float,
    ax=None,
    N: int = None,
    x_dataset="x0",
    y_dataset="y0",
    psi_dataset="psi",
    plot_boats=True,
    styles: dict = {},
) -> plt.axes:

    if ax is None:
        fig, ax = plt.subplots()

    standard_styles_ = standard_styles.copy()
    for label, df in dataframes.items():

        if label in styles:
            style = styles[label]

        else:

            if len(standard_styles) > 0:
                standard_style = standard_styles_.pop(0)
            else:
                standard_style = standard_styles_[0]

            style = {}
            style["style"] = standard_style

        track_plot(
            df=df,
            lpp=lpp,
            beam=beam,
            ax=ax,
            N=N,
            x_dataset=x_dataset,
            y_dataset=y_dataset,
            psi_dataset=psi_dataset,
            plot_boats=plot_boats,
            **style,
        )

    return ax


def track_plot(
    df,
    lpp: float,
    beam: float,
    ax=None,
    N: int = None,
    x_dataset="x0",
    y_dataset="y0",
    psi_dataset="psi",
    plot_boats=True,
    **plot_kwargs,
):

    if ax is None:
        fig, ax = plt.subplots()

    x = df[y_dataset]
    y = df[x_dataset]

    if N is None:
        s = np.sum(np.sqrt(x.diff() ** 2 + y.diff() ** 2))
        N = int(np.floor(s / lpp))

    # lines = ax.plot(x, y, **plot_kwargs)

    df.plot(x=y_dataset, y=x_dataset, **plot_kwargs, ax=ax)

    if plot_boats:
        _track_plot(
            time=df.index,
            x=np.array(df[x_dataset]),
            y=np.array(df[y_dataset]),
            psi=np.array(df[psi_dataset]),
            lpp=lpp,
            beam=beam,
            ax=ax,
            N=N,
            line_style="b-",
            alpha=1.0,
        )

    ax.set_xlabel("y0 [m]")
    ax.set_ylabel("x0 [m]")

    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_title("Track plot")
    return ax


def _track_plot(time, x, y, psi, lpp, beam, ax, N=7, line_style="y", alpha=1):

    indexes = np.linspace(0, len(time) - 1, N).astype(int)

    for i, index in enumerate(indexes):
        if i == 0:
            color = "g"
            alpha_ = 0.8
        elif i == (len(indexes) - 1):
            color = "r"
            alpha_ = 0.8
        else:
            color = line_style
            alpha_ = 0.5

        plot_ship(
            x[index],
            y[index],
            psi[index],
            lpp=lpp,
            beam=beam,
            ax=ax,
            color=color,
            alpha=alpha * alpha_,
        )


def plot_ship(x, y, psi, ax, lpp, beam, color="y", alpha=0.1):
    """Plot a simplified contour od this ship"""
    recalculated_boat = get_countour(x, y, psi, lpp=lpp, beam=beam)
    x = recalculated_boat[1]
    y = recalculated_boat[0]

    ax.plot(x, y, color, alpha=alpha)
    ax.fill(x, y, color, zorder=10, alpha=alpha)


def get_countour(x, y, psi, lpp, beam):
    # (Old Matlab boat.m)
    tt1 = lpp / 2
    tt2 = 0.9
    tt3 = beam / 4
    tt4 = 0.8
    tt5 = 3 * beam / 8
    tt6 = 0.6
    tt7 = beam / 2
    tt8 = 1.85 * beam / 4
    boat = np.matrix(
        [
            [
                tt1,
                tt2 * tt1,
                tt4 * tt1,
                tt6 * tt1,
                -tt4 * tt1,
                -tt2 * tt1,
                -tt1,
                -tt1,
                -tt2 * tt1,
                -tt4 * tt1,
                tt6 * tt1,
                tt4 * tt1,
                tt2 * tt1,
                tt1,
            ],
            [0, -tt3, -tt5, -tt7, -tt7, -tt8, -tt5, tt5, tt8, tt7, tt7, tt5, tt3, 0],
        ]
    )
    delta = np.array([[x], [y]])

    rotation = np.matrix([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])
    rows, columns = boat.shape
    rotated_boat = np.matrix(np.zeros((rows, columns)))
    for column in range(columns):
        rotated_boat[:, column] = rotation * boat[:, column]
    recalculated_boat = np.array(rotated_boat + delta)
    return recalculated_boat


test_type_xplot = {
    "Rudder angle": "delta",
    "Drift angle": "beta",
    "Circle": "r",
    "Circle + Drift": "v*r",
    "Rudder and drift angle": "delta",
    "resistance": "u",
    "Rudder and circle": "delta",
    "Thrust variation": "thrust",
}


def captive_plot(
    df_captive: pd.DataFrame,
    dofs=["fx", "fy", "mz"],
    styles=["-", ".", "o"],
    right=0.80,
    add_legend=True,
    **kwargs,
):

    df_captive = df_captive.copy()
    df_captive["v*r"] = df_captive["v"] * df_captive["r"]
    df_captive["beta"] = -np.arctan2(df_captive["v"], df_captive["u"])

    for test_type, df_ in df_captive.groupby(by="test type"):

        by_label = {}
        fig, axes = plt.subplots(ncols=len(dofs))

        for dof, ax in zip(dofs, axes):

            x_key = test_type_xplot.get(test_type, "V")

            if x_key == "u":
                plot_V(
                    df_=df_,
                    x_key=x_key,
                    styles=styles,
                    ax=ax,
                    dof=dof,
                    **kwargs,
                )
            else:
                for V, df_V in df_.groupby(by="V"):
                    plot_V(
                        df_=df_V,
                        x_key=x_key,
                        styles=styles,
                        ax=ax,
                        dof=dof,
                        **kwargs,
                    )

            ax.set_title(dof)
            ax.grid()
            ax.set_ylim((df_captive[dof].min(), df_captive[dof].max()))

            handles, labels = ax.get_legend_handles_labels()
            by_label.update(zip(labels, handles))
            ax.get_legend().set_visible(False)

        fig.suptitle(test_type, fontsize=16)

        if add_legend:
            fig.legend(by_label.values(), by_label.keys(), loc=7)
            fig.tight_layout()
            fig.subplots_adjust(right=right)
        else:
            fig.tight_layout()


def plot_V(df_, x_key, styles, ax, dof, **kwargs):

    for i, (item, df_item) in enumerate(df_.groupby(by="item")):

        if i < len(styles):
            style = styles[i]
        else:
            style = styles[-1]

        df_item.sort_values(by=x_key).plot(
            x=x_key,
            y=dof,
            style=style,
            label=item,
            ax=ax,
            **kwargs,
        )
