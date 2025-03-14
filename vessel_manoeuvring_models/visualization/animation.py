from .plot import track_plot
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from tqdm.notebook import tqdm

from matplotlib import animation
class NoLoopPillowWriter(animation.PillowWriter):	# Inherit PillowWriter
    def finish(self):
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps))  	# Not having 'loop' will run seq only once

def track_plot_anim(   
    df,
    lpp: float,
    beam: float,
    file_path =  "test.gif",
    ax=None,
    N: int = None,
    x_dfset="x0",
    y_dfset="y0",
    psi_dfset="psi",
    plot_boats=True,
    flip=True,
    time_window=[0, np.inf],
    start_color="g",
    stop_color="r",
    outline=False,
    equal=True,
    delta=False,
    dpi=200,
    speedup=10,
    fps=15,
    include_acceleration=False,
    include_velocity=False,
    dynamic_zoom=True,
    **plot_kwargs,):

    metadata = {'title':'animation'}
    
    #t_max=df.index[-1]/speedup
    #fps = N/t_max
    #fps_ = fps*speedup
    writer = NoLoopPillowWriter(fps=fps, metadata=metadata)

    ## Find a suitable size of plot:
    fig,ax=plt.subplots()
    fig.set_size_inches(10,5)

    ax = track_plot(df=df, lpp=lpp, beam=beam, N=10, flip=flip, ax=ax, include_acceleration=include_acceleration, include_velocity=include_velocity);
    ax.set_aspect('equal', 'box')
    figure = ax.get_figure()
    size = figure.get_size_inches()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()

    fig,ax=plt.subplots()
    fig.set_size_inches(size)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    xticks = np.arange(xlim[0],xlim[-1],5)

    dt = 1/fps*speedup
    ts = np.arange(df.index[0], df.index[-1],dt)

    N_boats = 10
    step = int(len(df)//N_boats)
    freeze_times = df.index[0::step]
    
    if include_acceleration or include_velocity:
        dynamic_zoom = False  # Does not work with annotation...

    
    if include_acceleration:
        x_key = 'u1d'
        y_key = 'v1d'
    else:
        x_key = 'u'
        y_key = 'v'
        
    
    vector_max = np.max(np.sqrt(df[x_key]**2 + df[y_key]**2))
    s_max = np.max(np.sqrt((df['x0'].max() - df['x0'].min())**2 + (df['y0'].max() - df['y0'].min())**2))
    scale_arrow = s_max / vector_max / 10


    with writer.saving(fig, file_path, dpi=dpi):
        for t in tqdm(ts):

            track_plot(df=df.loc[0:t], lpp=lpp, beam=beam, N=10, flip=flip, ax=ax, freeze_times=freeze_times, equal=False, delta=True, lw=0.5, alpha=1, 
                       include_acceleration=include_acceleration, include_velocity=include_velocity, scale_arrow=scale_arrow);

            ax.axis('off')
            ax.set_title('')
            ax.grid(False)
            ax.set_yticks([])
            ax.get_legend().set_visible(False)


            ax.set_xticks([])
            ax.set_xticklabels([])

            dynamic_zoom            
            
            if not dynamic_zoom:
                ax.plot(xlim,ylim,'.w')
            
            ax.set_aspect('equal', 'box')

            writer.grab_frame()

            ax.clear()