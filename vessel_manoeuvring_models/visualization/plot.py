import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from copy import deepcopy
import sympy as sp

standard_styles = ["b", "r", "g", "m", "c", "y"]


def plot(
    dataframes: dict,
    subplot=True,
    fig_size=(10, 10),
    styles: list = None,
    keys: list = None,
    ncols=2,
    time_window=[0, np.inf],
    zero_origo=True,
    sort_keys=True,
    units={},
    symbols={},
    include_label=True,
):
    if keys is None:
        keys = set()
        for label, df in dataframes.items():
            keys = keys | set(df.columns)

    if subplot:
        number_of_axes = len(keys)
        nrows = int(np.ceil(number_of_axes / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows)
        # fig.set_size_inches(fig_size)
        if ncols > 1 or nrows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

    plot_kwargs = {}

    styles = deepcopy(styles)
    
    if isinstance(styles, list):
        plot_kwargs = {
            label: {"style": style} for label, style in zip(dataframes.keys(), styles)
        }
    elif isinstance(styles, dict):
        styles = styles.copy()
        plot_kwargs = deepcopy(styles)

    standard_styles_ = standard_styles.copy()
    for key in dataframes.keys():
        if not key in plot_kwargs:
            plot_kwargs[key] = {}

        # if not "style" in plot_kwargs[key]:
        #    if len(standard_styles_) > 1:
        #        standard_style = standard_styles_.pop(0)
        #    else:
        #        standard_style = standard_styles_[0]
        #
        #    plot_kwargs[key]["style"] = standard_style

        if not "label" in plot_kwargs[key]:
            plot_kwargs[key]["label"] = key

    if sort_keys:
        iteration_keys = sorted(keys)
    else:
        iteration_keys = keys

    for i, key in enumerate(iteration_keys):
        if subplot:
            ax = axes[i]
        else:
            fig, ax = plt.subplots()

        unit = units.get(key, "")

        for label, df in dataframes.items():
            plot_kwarg = plot_kwargs.get(label, {})
            
            if not include_label:
                plot_kwarg['label']='__none__'

            if key in df:
                mask = (df.index >= time_window[0]) & (df.index <= time_window[1])
                df_ = df.loc[mask]
                x = df_.index

                if unit == "rad":
                    y = np.rad2deg(df_[key])
                else:
                    y = df_[key]

                if "style" in plot_kwarg:
                    _plot_kwarg = plot_kwarg.copy()
                    style = _plot_kwarg.pop("style")
                    ax.plot(x, y, style, **_plot_kwarg)
                else:
                    ax.plot(x, y, **plot_kwarg)

        ax.grid(True)

        if zero_origo:
            ylims = ax.get_ylim()
            ax.set_ylim(min(0, ylims[0]), max(0, ylims[1]))

        legend = ax.get_legend()
        if legend:
            legend.set_visible(False)

        symbol = symbols.get(key, key)
        if isinstance(symbol, sp.Symbol):
            symbol = sp.latex(symbol)
        if unit == "rad":
            unit = "deg"
        y_label = f"${symbol}$ [{unit}]" if unit != "" else f"${symbol}$"
        ax.set_ylabel(y_label)

    lines = [len(ax.lines) for ax in axes]
    i = np.argmax(lines)
    axes[i].legend()

    for ax in fig.axes[0:-ncols]:
        ax.set_xticklabels([])
        ax.set_xlabel("")

    for ax in fig.axes[-ncols:]:
        ax.set_xlabel("Time [s]")

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
    flip=False,
    time_window=[0, np.inf],
    include_label=True,
) -> plt.axes:
    styles = deepcopy(styles)

    if ax is None:
        fig, ax = plt.subplots()

    # standard_styles_ = standard_styles.copy()
    for label, df in dataframes.items():
        if label in styles:
            style = styles[label]

        else:
            #    if len(standard_styles_) > 1:
            #        standard_style = standard_styles_.pop(0)
            #    else:
            #        standard_style = standard_styles_[0]
            #
            style = {}
        #    style["style"] = standard_style

        if not "label" in style:
            style["label"] = label

        if not include_label:
            style["label"] = "__none__"
        
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
            flip=flip,
            time_window=time_window,
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
    flip=False,
    time_window=[0, np.inf],
    start_color="g",
    stop_color="r",
    outline=False,
    equal=True,
    delta=False,
    freeze_times: list=None,
    include_acceleration=False,
    include_velocity=False,
    scale_arrow=None,
    **plot_kwargs,
):
    if ax is None:
        fig, ax = plt.subplots()

    mask = (df.index >= time_window[0]) & (df.index <= time_window[1])
    df = df.loc[mask].copy()

    if flip:
        df_old = df.copy()
        df = df.copy()
        df[x_dataset] = -df_old[y_dataset]
        df[y_dataset] = df_old[x_dataset]
        df[psi_dataset] += np.deg2rad(90)

    x = df[y_dataset]
    y = df[x_dataset]

    if N is None:
        s = np.sum(np.sqrt(x.diff() ** 2 + y.diff() ** 2))
        N = int(np.floor(s / lpp))

    # lines = ax.plot(x, y, **plot_kwargs)

    df.plot(x=y_dataset, y=x_dataset, **plot_kwargs, ax=ax)
    
    if plot_boats:
        if delta:
            delta = df['delta'].values
        else:
            delta = None
        indexes = _track_plot(
            time=df.index,
            x=np.array(df[x_dataset]),
            y=np.array(df[y_dataset]),
            psi=np.array(df[psi_dataset]),
            lpp=lpp,
            beam=beam,
            ax=ax,
            N=N,
            line_style="b-",
            start_color=start_color,
            stop_color=stop_color,
            outline=outline,
            delta=delta,
            freeze_times=freeze_times,
            **plot_kwargs,
        )
    
    if include_acceleration:
        plot_accelerations(df=df.iloc[indexes], ax=ax, scale=scale_arrow)
        #plot_accelerations(df=df.iloc[[-1]], ax=ax)
        
        
    if include_velocity:
        plot_velocities(df=df.iloc[indexes], ax=ax, scale=scale_arrow)
    
    if flip:
        ax.set_xlabel("$x_0$ $[m]$")
        ax.set_ylabel("$y_0$ $[m]$")
    else:
        ax.set_xlabel("$y_0$ $[m]$")
        ax.set_ylabel("$x_0$ $[m]$")

    ax.grid(True)
    # ax.set_aspect("equal")
    if equal:
        ax.axis("equal")
    
    ax.set_title("Track plot")
    return ax


def _track_plot(
    time,
    x,
    y,
    psi,
    lpp,
    beam,
    ax,
    N=7,
    line_style="y",
    alpha=1,
    start_color="g",
    stop_color="r",
    outline=False,
    delta:float=None,
    freeze_times: list=None,
    **plot_kwargs,
):
    if freeze_times is None:
        if N == 1:
            indexes = [len(time) - 2]  # Only last if N=1
        else:
            indexes = np.linspace(0, len(time) - 1, N).astype(int)
    else:
        
        mask = np.isin(time,freeze_times)
        mask[-1] = True  # Always include last...
        indexes = np.arange(len(time))[mask]
    
    if 'color' in plot_kwargs:
        plot_kwargs = plot_kwargs.copy()
        plot_kwargs.pop('color')
    
    for i, index in enumerate(indexes):
        if i == 0:
            color = start_color
            alpha_ = 0.2

        if i == (len(indexes) - 1):
            color = stop_color
            alpha_ = 0.2
        else:
            color = line_style
            alpha_ = 0.2

        if not delta is None:
            delta_ = delta[index]
        else:
            delta_ = None
        
        plot_ship(
            x[index],
            y[index],
            psi[index],
            lpp=lpp,
            beam=beam,
            ax=ax,
            color=color,
            alpha=alpha * alpha_,
            outline=outline,
            delta=delta_,
            **plot_kwargs,
        )
        
    return indexes


def plot_ship(x, y, psi, ax, lpp, beam, color="y", alpha=0.1, outline=False, delta:float=None, **kwargs):
    """Plot a simplified contour od this ship"""
    
    x0=x
    y0=y
    
    recalculated_boat = get_countour(x, y, psi, lpp=lpp, beam=beam)
    x = recalculated_boat[1]
    y = recalculated_boat[0]

    if not "zorder" in kwargs:
        kwargs["zorder"] = 10

    if outline:
        outline_color = "k"
        outline_alpha = 1
        ax.plot(x, y, outline_color, alpha=outline_alpha, **kwargs)
    else:
        outline_color = color
        outline_alpha = alpha
        
        if 'style' in kwargs:
            kwargs=kwargs.copy()
            kwargs.pop('style')
        
        ax.plot(x, y, outline_color, alpha=outline_alpha, **kwargs)
        ax.fill(x, y, color, alpha=alpha, **kwargs)
        
    if not delta is None:
        ## Draw a rudder:
        x_r0 = x0-lpp*0.48*np.cos(psi)
        y_r0 = y0-lpp*0.48*np.sin(psi)
        #x_r0 = x0
        #y_r0 = y0
        angle = 2*delta + psi
        x_r1 = x_r0 -lpp/5*np.cos(angle)
        y_r1 = y_r0-lpp/5*np.sin(angle)
        ax.plot([y_r0,y_r1],[x_r0,x_r1], 'k-', lw=1)
        


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

def plot_accelerations(df:pd.DataFrame, ax, scale=None):
    plot_vectors(df=df, ax=ax, x_key='u1d', y_key='v1d', scale=scale)    

def plot_velocities(df:pd.DataFrame, ax, scale=None):
    plot_vectors(df=df, ax=ax, x_key='u', y_key='v', scale=scale)    


def plot_vectors(df:pd.DataFrame, ax, x_key='u', y_key='v', scale=None):
    
    if scale is None:
        vector_max = np.max(np.sqrt(df[x_key]**2 + df[y_key]**2))
        s_max = np.max(np.sqrt((df['x0'].max() - df['x0'].min())**2 + (df['y0'].max() - df['y0'].min())**2))
        scale = s_max / vector_max / 10   
    
    for i, row in df.iterrows():
        plot_vector(row=row, ax=ax, x_key=x_key, y_key=y_key, scale=scale)

def plot_vector(row:pd.Series, ax, x_key='u', y_key='v', scale=1):
    
    origin = np.array([row['y0'],row['x0']])
    
    dxy = np.array([row[x_key],row[y_key]])*scale
    
    xy = origin + dxy
    
    
    arrow(x0=origin[0],y0=origin[1],x2=xy[0],y2=xy[1], ax=ax, color='k')
    
    #ax.plot(origin[0],origin[1],'k.')

def arrow(x0,y0,x2,y2,ax, head_length=None, head_width=None, **plot_kwargs):


    # Calculate the angle of the arrow
    angle = np.arctan2(y2 - y0, x2 - x0)
    length = np.sqrt((y2 - y0)**2 + (x2 - x0)**2)
    
    # Define the arrow shape using plot function with specified origin and end point
    arrow_x = [x0, x2]
    arrow_y = [y0, y2]
    
    if not 'color' in plot_kwargs:
        plot_kwargs['color'] = 'b'
    
    # Plot the arrow shaft
    the_plot = ax.plot(arrow_x, arrow_y, **plot_kwargs)
    
    # Define the head of the arrow
    if head_length is None: 
        head_length = 0.1*length

    if head_width is None:
        head_width = 0.05*length
    
    # Calculate the coordinates of the arrow head
    hx1 = x2 - head_length * np.cos(angle - np.pi / 6)
    hy1 = y2 - head_length * np.sin(angle - np.pi / 6)
    hx2 = x2 - head_length * np.cos(angle + np.pi / 6)
    hy2 = y2 - head_length * np.sin(angle + np.pi / 6)


    
    # Plot the arrow head
    ax.plot([x2, hx1], [y2, hy1], **plot_kwargs)
    ax.plot([x2, hx2], [y2, hy2], **plot_kwargs)


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

    colors = ["r", "g", "b"]
    color_map = {}
    for V in df_captive["V"].unique():
        if len(colors) > 1:
            color = colors.pop(0)
        else:
            color = colors[0]

        color_map[V] = color

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
                    color=color_map[V],
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
                        color=color_map[V],
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


def plot_V(df_, x_key, styles, ax, dof, color, **kwargs):
    for i, (item, df_item) in enumerate(df_.groupby(by="item")):
        if i < len(styles):
            style = styles[i]
        else:
            style = styles[-1]

        style += color
        df_item.sort_values(by=x_key).plot(
            x=x_key,
            y=dof,
            style=style,
            label=item,
            ax=ax,
            **kwargs,
        )


def plot_parameters(parameters: pd.DataFrame, quantile_cuts=[0.6, 0.95]) -> plt.figure:
    """Plot regressed parameters (hydrodynamic derivatives) as bar plots.
    As the magnitute differs a lot, it is possible to split into many bar plots divided with
    "quantile_cuts"

    Parameters
    ----------
    parameters : pd.DataFrame
        must have "mean" and "std"

    quantile_cuts : list, optional
        Make barplots for these quantiles, by default [0.6, 0.95]

    Returns
    -------
    plt.figure
        figure with bar plots
    """

    cuts = [0] + quantile_cuts + [1]
    cuts = np.flipud(cuts)

    N = len(cuts) - 1
    fig, axes = plt.subplots(nrows=N)
    if N == 1:
        axes = [axes]

    # fig.set_size_inches(size_inches[0], N * size_inches[1])

    for i, ax in zip(range(N), axes):
        cut_start = cuts[i + 1]
        cut_stop = cuts[i]

        mask = (
            parameters["mean"].abs() > parameters["mean"].abs().quantile(cut_start)
        ) & (parameters["mean"].abs() <= parameters["mean"].abs().quantile(cut_stop))

        parameters.loc[mask].plot.bar(y="mean", yerr=parameters["std"], ax=ax)

    plt.tight_layout()
    return fig


def parameter_contributions(
    data_prime: pd.DataFrame, diff_eq, parameters: dict
) -> pd.DataFrame:
    X = diff_eq.calculate_features(data_prime)
    parameters = pd.Series(parameters)
    mask = parameters != 0
    parameters = parameters[mask].copy()
    keys = list(set(X.columns) & set(parameters.keys()))
    forces = X.multiply(parameters[keys]).dropna(how="all", axis=1)

    return forces

def join_contributions(df_contributions: pd.DataFrame, joins=["v", "r"]):

    df_contributions_joined = pd.DataFrame(index=df_contributions.index)

    for join in joins:
        mask = df_contributions.columns.str.contains(join)
        columns = df_contributions.columns[mask]
        new_column_name = "+".join(columns)
        df_contributions_joined[new_column_name] = df_contributions[columns].sum(axis=1)

    return df_contributions_joined

def plot_parameter_contributions(data: pd.DataFrame, model, regression):
    diff_eqs = {
        "X": regression.diff_eq_X,
        "Y": regression.diff_eq_Y,
        "N": regression.diff_eq_N,
    }

    parameters = model.parameters
    data_prime = model.prime_system.prime(data, U=data["U"])

    for dof, diff_eq in diff_eqs.items():
        forces = parameter_contributions(
            data_prime=data_prime, diff_eq=diff_eq, parameters=parameters
        )
        fig = px.line(forces, y=forces.columns, width=800, height=350, title=dof)
        display(fig)

