import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def track_plot(
    df,
    lpp: float,
    beam: float,
    ax=None,
    N: int = None,
    x_dataset="x0",
    y_dataset="y0",
    psi_dataset="psi",
    **plot_kwargs,
):

    if ax is None:
        fig, ax = plt.subplots()

    x = df[y_dataset]
    y = df[x_dataset]

    if N is None:
        s = np.sum(np.sqrt(x.diff() ** 2 + y.diff() ** 2))
        N = int(np.floor(s / lpp))

    lines = ax.plot(x, y, **plot_kwargs)
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
}


def captive_plot(
    df_captive: pd.DataFrame,
    dofs=["fx", "fy", "mz"],
    suffixes=[],
    styles=["-", ".", "o"],
    legends=["model"],
    right=0.80,
    add_legend=True,
    **kwargs,
):

    assert len(styles) > len(
        suffixes
    ), "not enough styles provided, each suffix must have a style"

    df_captive = df_captive.copy()
    df_captive["v*r"] = df_captive["v"] * df_captive["r"]
    df_captive["beta"] = -np.arctan2(df_captive["v"], df_captive["u"])

    for test_type, df_ in df_captive.groupby(by="test type"):

        by_label = {}
        fig, axes = plt.subplots(ncols=len(dofs))

        for dof, ax in zip(dofs, axes):

            x_key = test_type_xplot[test_type]
            df_.sort_values(by=x_key).plot(
                x=x_key, y=dof, style=styles[0], label=legends[0], ax=ax, zorder=10
            )

            for i, suffix in enumerate(suffixes):
                dof_vct = f"{dof}{suffix}"

                try:
                    label = legends[i + 1]
                except:
                    label = suffix[1:]

                df_.sort_values(by=x_key).plot(
                    x=x_key,
                    y=dof_vct,
                    style=styles[i + 1],
                    label=label,
                    ax=ax,
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
