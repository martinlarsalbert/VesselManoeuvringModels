from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_time_series(data:pd.DataFrame, keys:list, units={}, height=800, width=1000, title_text="Time series"):

    rows = len(keys)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True)

    yaxis_titles = []
    for row,key in zip(range(1,rows+1),keys):
        
        if isinstance(key, tuple):
            for sub_key in key:
                _plot(fig=fig, units=units, key=sub_key, data=data,row=row)
        else:
            _plot(fig=fig, units=units, key=key, data=data,row=row)
        
        unit=units.get(key,'')
        if unit is 'rad':
            unit='deg'
        
        if len(unit) > 0:
            yaxis_title = f"{key} [{unit}]"
        else:
            yaxis_title = f"{key} "
            
        yaxis_titles.append(yaxis_title)
          
    fig.update_layout(height=height, width=width, title_text=title_text,yaxis_title=yaxis_title)
    
    fig['layout']['yaxis']['title'] = yaxis_titles[0]
    for i,yaxis_title in enumerate(yaxis_titles):
        fig['layout'][f'yaxis{i+1}']['title']=yaxis_title 
        
    return fig

def _plot(fig, units, key, data, row):
    
    unit=units.get(key,'')
        
    if unit is 'rad':
        y=np.rad2deg(data[key])
        unit='deg'
    else:
        y=data[key]
            
    fig.append_trace(go.Scatter(
        x=data.index,
        y=y,
        name=key,
    ), row=row, col=1)
