
"""
Created on Fri Jun 03 13:45:41 2016

@author: laurajanem

This module does some plotting things

"""

def spectra(xdata, ydata, plot_params={},current_axis=None):
    if len(plot_params)==0:
        plot_params = get_default_plot_params('spectra')
    fig, line_handles, current_axis = plot(xdata,ydata,plot_params,current_axis)
    return fig, line_handles, current_axis

def fids(xdata, ydata, plot_params={},current_axis=None):
    if len(plot_params)==0:
        plot_params = get_default_plot_params('fid')
    fig, line_handles, current_axis = plot(xdata,ydata,plot_params,current_axis)
    return fig, line_handles, current_axis
    

def plot(xdata, ydata, plot_type='', plot_params, current_axis=None):
    """
    
    Parameters
    ----------
    xdata : array_like
            Data to plot on x-axis
    ydata : array_like
            Data to plot on y-axis
    plot_type : string-optional
                Type of plot to generate. Determines some default plotting parameters, if they are specified.
    plot_params : dictionary-optional
                  Dictionary of parameters to use for generating the graph and formatting the axes. 
                  If plot_params = {}, default parameters are used.
    current_axis : axis handle-optional
                   If not empty, add the plot to this axis, instead of generating a new figure + axis
                  
    Returns
    -------
    current_axis : axis handle
                   Handle of the axis the line plots were added to
     
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
#    import seaborn
    
    # Check on the shape of the input data
    if ydata.shape[0] != xdata.shape[0] and ydata.shape[0] != xdata.shape[0]:
        # Throw an error
        pass
    elif ydata.shape[0] != xdata.shape[0]:
        # Transpose the data
        ydata = ydata.T

        
    # Get the plot parameters
    if len(plot_params)==0:
        # Use defaults
        plot_params = get_default_plot_params(plot_type)
    
    plt.ioff()
    if current_axis is not None:
        # Add the plot to a previously specified axis
        line_handles = current_axis.plot(xdata,ydata)
        fig = plt.gcf()
    else:
        fig=plt.figure()  
        current_axis = fig.add_subplot(1,1,1)
        line_handles = current_axis.plot(xdata,ydata)

    
    if plot_params['overlay_average']:
        ydata_mean = np.average(ydata,axis=1)
        current_axis.plot(xdata,ydata_mean,linewidth=3,color='k')
        line_handles = current_axis.get_lines()
    
    if plot_params['save_fig']:
        # Save the figure
        pass

    apply_plot_params(plot_params)

    if plot_params['autoclose']:
        # Close the figure before exiting the function
        plt.close(fig)
        fig = []
        line_handles = []
        current_axis = []

    
    return fig, line_handles, current_axis
    

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
                    Maximum values of the average
            stuff
           
    
    
    """
    c_idx = 0
    max_chan_mags = 1
    
    import numpy as np
    # Determine how many channels there are, based on the shape of the data channel index
    
    max_chan_mags = np.max(np.average(np.abs(data),axis=0),axis=1)      
   
    return c_idx, max_chan_mags

    

def get_default_plot_params(plot_type=''):    
    """
    Return the default parameters used for figure generation.
    
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
        'title':'',
        'xlabel':'',
        'ylabel':'',
        'xlim':None,
        'ylim':None,
        'grid':'on',
        'overlay_average':False,
        'save_fig': False,
        'output_fig_path':'',
        'output_fig_type':'png',
        'autoclose': False,
        'suppress_fig':False,
        'interactive': True,
        'backend':'',
        'reverse_x': False
        }
    
    # Add more plot-type specific parameters to the dictionary   
    # Note: This is a good spot                
    if plot_type is 'spectra':
        default_plot_params.update({'reverse_x':True})
    elif plot_type is 'fid':
        default_plot_params['reverse_x']=False
    else:
        # Note: Here is a good spot to add customized plotting parameters
        pass
            
                       
    return default_plot_params
    

def apply_plot_params(plot_params):
    """Apply parameters to current axis."""
    
    import matplotlib.pyplot as plt
    
    if plot_params['xlim'] is not None: 
        plt.xlim(plot_params['xlim'])  
        
        if plot_params['reverse_x'] and plot_params['xlim'][0] < plot_params['xlim'][1]:
            plt.gca().invert_xaxis()
            
    elif plot_params['reverse_x']:
        plt.gca().invert_xaxis()
    
    if plot_params['ylim'] is not None: plt.ylim(plot_params['ylim'])
        
    plt.grid(plot_params['grid'])
    plt.title(plot_params['title'])
    plt.xlabel(plot_params['xlabel'])
    plt.ylabel(plot_params['ylabel'])
    
    
    if plot_params['save_fig']:
        # Save the figure to the specified location
        plt.savefig(plot_params['output_fig_path'],transparent=True)
        pass
    
    if plot_params['suppress_fig'] is False:
        plt.draw()
        plt.show()
    
    plt.ion()
    
    return []
    




