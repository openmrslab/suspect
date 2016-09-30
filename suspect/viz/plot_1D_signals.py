
"""
Created on Fri Jun 03 13:45:41 2016

@author: laurajanem

This module does some plotting things

"""


def plot(xdata, ydata, plot_type='', plot_params={}, ax=None):
    """Generic function for plotting any kind of MRS data.

    Uses matplotlib's pyplot for plotting but automatically applies MRS-data specific formatting parameters.
    
    Parameters
    ----------
    xdata : array_like
        Data to plot on x-axis
    ydata : array_like
        Data to plot on y-axis
    plot_type : string-optional
        Type of plot to generate. Determines some default plotting parameters, if they are specified.
        Currently implemented: ['spectrum', 'fids']
    plot_params : dictionary-optional
        Dictionary of parameters to use for generating the graph and formatting the axes.
        If plot_params = {}, default parameters are used.
    ax : axis handle-optional
        If not empty, add the plot to this axis, instead of generating a new figure + axis
                  
    Returns
    -------
    fig_handle : figure handle
        Handle of the figure the data was added to
    line_handles : 2D line object handle(s)
        List of handles to each of the line objects added to the current axis
    ax : axes handle
        Handle of the axes data was added to
     
    """
    import numpy as np
    from cycler import cycler
    import matplotlib.pyplot as plt
    
    # Check on the shape of the input data
    if ydata.shape[0] != xdata.shape[0] and ydata.shape[1] != xdata.shape[0]:
        # Throw an error
        pass
    elif ydata.shape[0] != xdata.shape[0]:
        # Transpose the data
        ydata = ydata.T

    # Get the plot parameters. Start with defaults, and then update to include any 
    # changes passed in to the function
    params = get_default_plot_params(plot_type)
    params.update(plot_params)

    if ax is None:
        if params['suppress_fig']:
            # Suppress drawing the figure
            plt.ioff()
        fig_handle = plt.figure(figsize=params['figsize'], facecolor='w')
        ax = fig_handle.add_subplot(1,1,1)

    # Set the color order cycle that will be applied to any lines added to the current axis        
    ax.set_prop_cycle(cycler('color',params['color_order']))
    
    if params['use_colormap'] and len(ydata.shape) > 1:
        # Override the color order using the specified color map
        # Distribute colormap colors evenly spaced for all the lines in the plot
    
        # Get the colors from the selected colormap to use for color cycling
        cmap = plt.get_cmap(params['colormap'])
        co = [cmap(ii) for ii in np.linspace(0.0, 0.9, ydata.shape[1])]
        ax.set_prop_cycle(cycler('color', co))
    
    # Add data to the axis
    ax.plot(xdata,ydata)  

    # Apply the plot parameters
    ax = apply_plot_params(params,ax)
    
    if params['overlay_average']:
        # Check on the dimensions of the data
        if len(ydata.shape)>1:
            ydata_mean = np.average(ydata, axis=1)
            ax.hold('on')
            ax.plot(xdata, ydata_mean, linewidth=2, color=params['overlay_average_color'])

        else:
            print('Ignoring ''overlay_average'', ydata is 1D')
    
    plt.tight_layout()              
    
    if params['save_fig']:
        # Save the figure to the specified location
        plt.savefig(params['output_fig_path'],transparent=True)

    # Turn plot interactivity back on in case it was turned off
    plt.ion()
    fig_handle = plt.gcf()
    line_handles = ax.get_lines()
    
    if params['autoclose']:
        # Close the figure containing the current axes
        plt.close(ax.figure)
        fig_handle = []
        line_handles = []
        ax = []

    return fig_handle, ax, line_handles
    

def apply_plot_params(plot_params,ax):
    """Apply parameters to current axis.

    Parameters
    ----------

    Returns
    -------


    """

    import matplotlib.pyplot as plt

    if plot_params['xlim'] is not None: 
        ax.set_xlim(plot_params['xlim'])
        
        if plot_params['reverse_x'] and plot_params['xlim'][0] < plot_params['xlim'][1]:
            ax.invert_xaxis()            
            
    elif plot_params['reverse_x']:
        ax.invert_xaxis()
    
    if plot_params['ylim'] is not None: ax.set_ylim(plot_params['ylim'])
        
    ax.grid(plot_params['grid'])
    ax.set_title(plot_params['title'], fontsize=plot_params['fontsize'], y=1.02)
    ax.set_xlabel(plot_params['xlabel'], fontsize=plot_params['tick_fontsize'])
    ax.set_ylabel(plot_params['ylabel'], fontsize=plot_params['tick_fontsize'])
    ax.set_axis_bgcolor(plot_params['axis_bg_color'])
    
    # Apply some properties to the line(s)
    line_handles = ax.get_lines()
    plt.setp(line_handles, lw=plot_params['line_width'], ls=plot_params['line_style'],
             marker=plot_params['marker'])
    
    if plot_params['line_color'] is not 'default':
        plt.setp(line_handles,'color', plot_params['line_color'])
    
    # Autoscale data on the y-axis to reflect changes to the x-axis
    if plot_params['autoscale_y']:
        ax = autoscale_y(ax, margin=0.05)
    
    if plot_params['use_sci_format_yaxis']:
        # Use scientific notation for the y-axis labels
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    
    if plot_params['use_sci_format_xaxis']:
        # Use scientific notation for the x-axis labels
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    
    ax.tick_params(labelsize=plot_params['tick_fontsize'])
        
    return ax


def suggest_channel(data):
    """
    If the use-case is plotting multiple averages collected from the same channel (coil)
    on the same axis, this function suggests a channel to use for plotting raw averages
    based on relative maximum magnitudes of the available channel data.
    
    Parameters
    ----------
    data : array_like
        Data array. Assumed to be formatted as [averages, channels, data_points]
    
    Returns
    -------
    c_idx : integer
        Index of suggested channel to plot
    max_chan_mags : array_like
        Maximum values of the average stuff

    """
    # c_idx = 0
    # max_chan_mags = 1
    
    import numpy as np
    # Determine how many channels there are, based on the shape of the data channel index
    
    max_chan_mags = np.max(np.average(np.abs(data), axis=0), axis=1)
    idx = np.argsort(max_chan_mags)
    c_idx = idx[-1]
   
    return c_idx, max_chan_mags


# Define some defaults for plotting different kinds of figs

def get_default_plot_params(plot_type=''):    
    """Return the default parameters used for figure generation.
    
    Parameter
    ---------
    plot_type : string-optional
        Defines some plot-specific parameters to use by default
                
    Returns
    -------
    default_plot_params : dictionary
        Dictionary key-value pairs used to define plotting parameters

    """   
    
    default_plot_params = {
    
        'autoclose': False,  
        'autoscale_y': True,
        'axis_bg_color': 'w',
        'colormap': 'summer',
        'color_order': ['#0072b2', '#d55e00', '#009e73', '#cc79a7', '#f0e442', '#56b4e9'],  # Seaborn 'colorblind'
        'figsize': (8, 6),  # Inches
        'fontsize': 14,
        'grid': 'on',
        'interactive': True,
        'line_color': 'default',
        'line_style': '-',
        'line_width': 2.0,
        'marker': None,
        'output_fig_path': 'test',
        'output_fig_type': 'png',
        'overlay_average': False,
        'overlay_average_color': '#424949',
        'reverse_x': False,
        'save_fig': False,    
        'suppress_fig': False,
        'tick_fontsize': 12,
        'title':'',
        'use_colormap': False,
        'use_sci_format_xaxis': False,
        'use_sci_format_yaxis': True,
        'xlabel': '',
        'xlim': None,
        'ylabel': '',
        'ylim': None,
        
        }
        
    # Some other good choices for color order!
    # 'color_order': ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974', '#64b5cd']
    # Based on a Seaborn 'deep' color palette

    # 'color_order': ['#4878cf', '#6acc65', '#d65f5f', '#b47cc7', '#c4ad66', '#77bedb'] #Seaborn 'muted'
    # 'color_order': ['#92c6ff', '#97f0aa', '#ff9f9a', '#d0bbff', '#fffea3', '#b0e0e6'] #Seaborn 'pastel'
    
    # Add more default plot-type specific parameters to the dictionary                  
    if plot_type is 'spectrum':  # Best for plotting 1-5 lines on the same axis
        default_plot_params.update({'reverse_x': True})
        
    elif plot_type is 'fid':  # Best for plotting 1-5 lines on the same axis
        default_plot_params.update({'reverse_x': False})
        
    elif plot_type is 'spectra':  # Best for plotting >5 lines on the same axis
        default_plot_params.update({'reverse_x': True, 'line_width': 1.0, 'overlay_average': True})
        
    elif plot_type is 'fids':  # Best for plotting >5 lines on the same axis
        default_plot_params.update({'reverse_x': False,'line_width': 1.0, 'overlay_average': True})
        
    else:
        # Note: Here is a good spot to add customized plotting parameters for specific kinds of plots
        pass

    return default_plot_params


def autoscale_y(ax, margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.

    Parameters
    ----------
    ax : matplotlib axes object
    margin : float
        the fraction of the total height of the y-data to pad the upper and lower ylims

    Returns
    -------
    ax : matplotlib axes object
        With y-axis rescaled

    """

    # Thanks to
    # http://stackoverflow.com/questions/29461608/matplotlib-fixing-x-axis-scale-and-autoscale-y-axis
    # for inspiring this!
    
    import numpy as np

    def get_bottom_top(line, lo, hi):
        xd = line.get_xdata()
        yd = line.get_ydata()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        if len(y_displayed) == 0:
            # No plotted data is inside the xlims
            return None, None
        else:
            h = np.max(y_displayed) - np.min(y_displayed)
            bot = np.min(y_displayed) - margin*h
            top = np.max(y_displayed) + margin*h
            return bot,top

    lines = ax.get_lines()
     
    lo, hi = ax.get_xlim()
    
    # Do a quick check to see if the x-axis has been inverted
    if lo > hi:  # Reverse them for now
        lo, hi = hi, lo
        
    # Initialize limits
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line,lo,hi)
        if new_bot is not None:
            if new_bot < bot:
                bot = new_bot
            if new_top > top:
                top = new_top

    # Check to see if it is appropriate to change the boundaries
    if bot != np.inf and top != -np.inf:
        ax.set_ylim(bot, top)
    
    return ax
