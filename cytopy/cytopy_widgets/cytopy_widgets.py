"""
This is a GUI wrapper for CytoPy, allowing to apply batch-gating 
to a large number of samples semi-automatically.

Copyright 2021 Eleven TX

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import logging
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1.5)

import ipywidgets as widgets
from ipywidgets import interactive, interact, Box, HBox, Layout, VBox, fixed, Checkbox, Layout, Text, Label
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from ipywidgets.widgets.widget_output import clear_output
import string
import random

from cytopy.data.experiment import Experiment
from cytopy.data.gating_strategy import GatingStrategy, EllipseGate, ThresholdGate, PolygonGate
from cytopy.data.geometry import probablistic_ellipse


def generate_random_name(N=20):
    '''
    Generate a random string with N letters
    '''
    gate_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
    return gate_name



def get_experiment_channels(experiment: Experiment):
    """
    Returns a list of all the channels in a specific experiment
    
    Parameters
    ----------
    experiment: Experiment
        An experiment object
    """
    
    channels = None
    for sample_id in experiment.list_samples():
        sample = experiment.get_sample(sample_id)
        df_sample = sample.load_population_df('root', transform=None)        
        sample_channels = df_sample.columns
        if channels is not None:
            assert len(sample_channels) == len(channels), \
            'not all samples in the experiment have the same channels'
            assert np.all(sample_channels == channels), \
            'not all samples in the experiment have the same channels'
        channels = sample_channels
    return list(channels)




class FACS_Gater:
    """
    This class acts as a wrapper to the CytoPy package.
    It basically applies a specific gate to a set of samples
    (refitting the gate for each sample).
    
    Parameters
    ----------
    experiment: Experiment
        A CytoPy experiment object
    parent_population: str
        The gating parent population
    sampling: dict (optional)
        Options for downsampling data prior to application of gate.
        See cytopy.flow.sampling for details.
    plot_bins: int or str (optional)
        How many bins to use for 2D histogram.
        See cytopy.flow.plotting.flow_plot for details
    autoscale: bool (default=True)
        Allow matplotlib to calculate optimal view
        (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.autoscale.html)
    """

    def __init__(self,
                 experiment: Experiment,
                 parent_population: str,
                 sampling: dict or None = None,
                 plot_bins: int or None = None,
                 autoscale: bool = False
                 ):
        
        #save constructor arguments
        self.experiment = experiment        
        self.sampling = sampling
        self.plot_bins = plot_bins
        self.autoscale = autoscale
        self.parent_population = parent_population
        
        #initialize gate and gate parameters
        self.dict_sampleid_to_gates = {}
        self.sample_ids = None
        self.gate = None
        
        #initialize gate and plotting parameters
        self.gating_params = {
            'transform_x': None,
            'transform_y': None,
            'x': None,
            'y': None
        }
            
        self.plot_params = {
            'xmin':None,
            'xmax':None,
            'ymin':None,
            'ymax':None
        }
        
        #store all sample_ids in this experiment
        self.experiment_sampleids = set(experiment.list_samples())
        
    def update_params(self,
                      x: str,
                      transform_x: str,
                      y: str or None = None,
                      transform_y: str or None = None,
                      xmin: int or None = None,
                      xmax: int or None = None,
                      ymin: int or None = None,
                      ymax: int or None = None
                     ):
        """
        Update the gating and plotting parameters
        
        Parameters
        ----------
        x: str
            The x axis channel
        transform_x: str
            The transformation to apply on the x-axis channel.
            Please see cytopy.flow.plotting.gate for details
        y: str (optional)
            The y axis channel (can be left empty for threshold 1d gates)
        transform_y: str (optional)
            The transformation to apply on the y-axis channel.
            Please see cytopy.flow.plotting.gate for details
        xmin: int (optional)
            The minimum x axis value to show in the plots
        xmax: int (optional)
            The maximum x axis value to show in the plots
        ymin: int (optional)
            The minimum y axis value to show in the plots
        ymax: int (optional)
            The maximum y axis value to show in the plots            
        """
            
        self.gating_params['transform_x'] = transform_x
        self.gating_params['transform_y'] = transform_y
        self.gating_params['x'] = x
        self.gating_params['y'] = y
        
        self.plot_params['xmin'] = xmin
        self.plot_params['xmax'] = xmax
        self.plot_params['ymin'] = ymin
        self.plot_params['ymax'] = ymax
        
            
    def ellipse_gate(self,
                     gate_name: str,
                     target_pop_name: str,
                     method: str = 'GaussianMixture',
                     n_clusters: int = 1,
                     conf:float = 0.8,
                     centroid: (float, float) or None = None,
                     angle: float or None = None,
                     width: float or None = None,
                     height: float or None = None
                    ):
        """
        Apply an ellipse gate on all of the samples
        
        parameters
        ----------
        gate_name: str
            The name of the gate
        target_pop_name: str
            The target population name
        method (default='GaussianMixture'):
            The Ellipse estimation method.
            see cytopy.data.gate for details
        n_clusters: int (default=1)
            The number of clusters (currently it must be 1)
        conf: float (default=0.8)
            The confidence level (i.e. the size of ellipes), must be between 0 and 1
        centroid: (float, float) (optional)
            The centroid of the ellipse (only required if method == 'manual')
        angle: float (optional)
            The angle of the ellipse (only required if method == 'manual')
        width: float (optional)
            The width of the ellipse (only required if method == 'manual')
        height: float (optional)
            The height of the ellipse (only required if method == 'manual')            
        """
            
        self._check_gating_params()
        method_kwargs = {'n_components': n_clusters, 'conf': conf}
        if method == 'manual':
            assert angle is not None
            assert width is not None
            assert height is not None
            assert centroid is not None
            assert conf is None
            method_kwargs['angle'] = angle
            method_kwargs['width'] = width
            method_kwargs['height'] = height
            method_kwargs['centroid'] = centroid
                         
        #define a population names dictionary
        assert n_clusters==1, 'Only n_clusters==1 is currently supported'
        dict_popnames = {'A':target_pop_name}

        #define a gate
        self.gate = EllipseGate(gate_name=gate_name,
                                   parent=self.parent_population,
                                   x=self.gating_params['x'],
                                   y=self.gating_params['y'],
                                   transform_x=self.gating_params['transform_x'],
                                   transform_y=self.gating_params['transform_y'],
                                   sampling=self.sampling,
                                   method=method,
                                   method_kwargs=method_kwargs)
        self._apply_gate(dict_popnames=dict_popnames, legend_delta_y=0)


    def threshold_gate(self,
                       gate_name: str,
                       method: str = 'density',
                       q: float or None = None,
                       x_threshold: float or None = None,
                       y_threshold: float or None = None,
                       bandwidth: float or None = None
                      ):
        """
        Apply a 1d or 2d threshold gate on all of the samples
        
        Parameters
        ----------
        gate_name: str
            The name of the gate
        method: str
            The gating method.
            Please see cytopy.data.gate for details
        q: float (optional)
            The desired thresholding quantile
            (only relevant if method=='quantile')
        x_threshold: float (optional)
            The threshold to apply on the x-axis channel
            (only required if method=='manual')
        y_threshold: float (optional)
            The threshold to apply on the y-axis channel
            (only required if method=='manual')
        bandwidth: float (optional)
            The density estimation bandwidth.
            Please see cytopy.data.gate for details
        """
            
        self._check_gating_params()
        method_kwargs = {}
        create_plot_kwargs = {}
        
        if bandwidth is not None:
            method_kwargs['bw'] = bandwidth
            create_plot_kwargs['bw'] = bandwidth
        
        if method == 'density':
            pass
        elif method == 'quantile':
            assert q is not None
            method_kwargs['q'] = q
        elif method == 'manual':
            assert x_threshold is not None
            if self.gating_params['y'] is not None: assert y_threshold is not None
            method_kwargs['x_threshold'] = x_threshold
            method_kwargs['y_threshold'] = y_threshold
        else:
            raise ValueError('unsupported method: %s'%(method))
        
        self.gate = ThresholdGate(gate_name=gate_name,
                                   parent=self.parent_population,
                                   x=self.gating_params['x'],
                                   y=self.gating_params['y'],
                                   transform_x=self.gating_params['transform_x'],
                                   transform_y=self.gating_params['transform_y'],
                                   sampling=self.sampling,
                                   method=method,
                                   method_kwargs=method_kwargs
                                  )
        
        self._apply_gate(create_plot_kwargs=create_plot_kwargs, legend_delta_y=0.2)

        
    def polygon_gate(self,
                     gate_name: str,
                     method: str = 'MiniBatchKMeans',
                     n_clusters: int = 2,
                     random_state: int = 111,
                     batch_size: int = 1000
                    ):
            
        """
        Apply a polygon gate on all of the samples
        
        Parameters
        ----------
        gate_name: str
            The name of the gate
        method: str
            The gating method.
            Please see cytopy.data.gate for details
        n_clusters: int (default=2)
            The number of clusters
        random_state: int (default=111)
            A random initial state for the random number generator
        batch_size: int (default=1000)
            The batch size of the gating algorithm
        """
             
        self._check_gating_params()
        method_kwargs = {}
        method_kwargs['n_clusters'] = n_clusters
        method_kwargs['batch_size'] = batch_size
        method_kwargs['random_state'] = random_state
        self.gate = PolygonGate(gate_name=gate_name,
                                   parent=self.parent_population,
                                   x=self.gating_params['x'],
                                   y=self.gating_params['y'],
                                   transform_x=self.gating_params['transform_x'],
                                   transform_y=self.gating_params['transform_y'],
                                   sampling=self.sampling,
                                   method=method,
                                   method_kwargs=method_kwargs
                                  )
        self._apply_gate(legend_delta_y=0.07*(n_clusters-1))

            
    def _check_gating_params(self):
        """
        Make sure that gating parameters were specified
        """
        for c in ['x']:
            assert self.gating_params[c] is not None, \
                'please call update_params(...) first'
            
            
    def _apply_gate(self,
                    legend_delta_y: float,
                    dict_popnames: dict or None = None,
                    create_plot_kwargs: dict or None = None
                   ):
        """
        Apply the object gate (specified in self.gate) to all
        sample_ids that were specified for gating
        
        parameters
        ------------
        dict_popnames: dict (optional)
            a dictionary mapping automatically-generated
            population names to new population names
        create_plot_kwargs: dict (optional)
            A dictionary of arguments for the constructor
            of the CytoPy FlowPlot object
        
        """
        self._check_gating_params()
        assert self.sample_ids is not None, 'no sample ids specified'
        
        #prepare a figure
        nsamples = len(self.sample_ids)
            
        ####Code for 2-columns view
        nrows = nsamples//2 + (1 if nsamples%2 == 1 else 0)
        ncols = (1 if nsamples==1 else 2)
        figsize = (11, 6) if nsamples==1 else (11, nrows*6)
        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

#         ####Code for 1-columns view
#         nrows = nsamples
#         ncols = 1
#         figsize = (11, nrows*6)
#         fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)
        
        if ncols==1 and nrows==1: axs = np.array([[axs]])
        elif ncols==1 or nrows==1: axs = np.array([axs])
        if create_plot_kwargs is None: create_plot_kwargs = {}
        create_plot_kwargs['bins'] = self.plot_bins
        create_plot_kwargs['autoscale'] = self.autoscale
        plot_kwargs = {'legend_kwargs':
                       {
                        'loc':'lower left',
                        'bbox_to_anchor':(0.99, 0.1), 
                        'fontsize':'xx-small'
                       }
                      }

        #iterate over sample_ids and plot the gate fitted to each sample_id
        iter_tqdm = tqdm(
                         zip(axs.flatten(), self.sample_ids),
                         total=nsamples,
                         desc='Creating gates'
                        )
        for ax, sample_id in iter_tqdm:
            
            #fit the gate on this sample_id
            gates = GatingStrategy(name=generate_random_name())
            gates.load_data(experiment=self.experiment, sample_id=sample_id)
            create_plot_kwargs['ax'] = ax
            
            #Fit the gate
            self.gate.reset_gate()
            gates.preview_gate(self.gate,
                               create_plot_kwargs=create_plot_kwargs,
                               plot_gate_kwargs=plot_kwargs,
                               plot=False)
            #ax.clear()

            #rename gate populations
            gate_populations = [c.name for c in self.gate.children]
            dict_labels_new = {}
            if dict_popnames is None: dict_popnames = {}
            for pop in gate_populations:
                if self.parent_population == 'root':
                        new_pop_name = dict_popnames.get(pop, pop)
                else:
                        new_pop_name = '%s->%s'%(
                            self.parent_population, dict_popnames.get(pop, pop)
                        )
                dict_labels_new[pop] = new_pop_name
            self.gate.label_children(dict_labels_new)
            gate_populations = [c.name for c in self.gate.children]

            #save the fitted GatingStrategy object in the dictionary
            self.dict_sampleid_to_gates[sample_id] = gates

            #delete existing populations with the same names if there are any
            pops_to_delete = [p for p in gate_populations if p in gates.list_populations()]
            gates.delete_populations(pops_to_delete)
            
            #apply the gate (finally!)
            gates.apply_gate(self.gate,
                                  plot=True,
                                  add_to_strategy=True,
                                  create_plot_kwargs=create_plot_kwargs,
                                  plot_gate_kwargs=plot_kwargs,
                                  hyperparam_search=False,
                                  fda_norm=False,
                                  verbose=False
                                 )
                
            #add title of sample_id
            population_size = \
                gates.filegroup.get_population(population_name=self.parent_population).n
            title = '%s\n%d cells'%(sample_id, population_size)
            ax.set_title(title, fontsize='xx-small')
            
            #change axis labels
            xlabel = str(self.gating_params['x'])
            if self.gating_params['transform_x'] is not None:
                xlabel += ' (%s)'%(self.gating_params['transform_x'])
            ylabel = str(self.gating_params['y'])
            if self.gating_params['transform_y'] is not None:
                    ylabel += ' (%s)'%(self.gating_params['transform_y'])
            ax.set_xlabel(xlabel, fontsize='x-small')
            ax.set_ylabel(ylabel, fontsize='x-small')
            
            #change ticks font size
            ax.tick_params(axis='both', which='major', labelsize='small')

            #make sure that the tick labels aren't rotated
            plt.setp(ax.get_xticklabels(), rotation=0)
            
            #move the legend to the bottom
            leg = ax.get_legend()
            #legend_bbox = leg.get_window_extent()
            leg.set_bbox_to_anchor((-0.1, -0.1-legend_delta_y), transform=ax.transAxes)
            
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top') 
            
        #enforce uniform limits
        x_min = min([ax.get_xlim()[0] for ax in axs.flatten()])
        y_min = min([ax.get_ylim()[0] for ax in axs.flatten()])
        x_max = max([ax.get_xlim()[1] for ax in axs.flatten()])
        y_max = max([ax.get_ylim()[1] for ax in axs.flatten()])
        if self.plot_params['xmin'] is not None: x_min = self.plot_params['xmin']
        if self.plot_params['ymin'] is not None: y_min = self.plot_params['ymin']
        if self.plot_params['xmax'] is not None: x_max = self.plot_params['xmax']
        if self.plot_params['ymax'] is not None: y_max = self.plot_params['ymax']
        for ax in axs.flatten():
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
    
    def list_populations(self):
        """
        Return a list of all of the populations that were created
        """
        if len(self.dict_sampleid_to_gates) == 0: return ['root']
        gates = next(iter(self.dict_sampleid_to_gates.values()))
        return gates.list_populations()
    
    def list_gate_populations(self):
        """
        Returns a list of the gate populations for the current gate
        """
        return [c.name for c in self.gate.children]
    
    def save_populations(self):
        """
        Save all created populations to the DB
        """
        assert len(self.dict_sampleid_to_gates) > 0, 'no gated were fitted yet'
        set_sampleids = set(self.sample_ids)
        assert len(set_sampleids) == len(self.dict_sampleid_to_gates)
        for sample_id, gates in self.dict_sampleid_to_gates.items():
            assert sample_id in set_sampleids
            gates.save(save_strategy=False, save_filegroup=True)
            
    def update_sample_ids(self,
                          sample_ids: List[str]
                         ):
        """
        Specify the set of sample_ids that the object is gating
        
        Parameters
        ----------
        sample_ids: List[str]
            The list of sample ids that the gating will work on
        """
        assert all([(s in self.experiment_sampleids) for s in sample_ids])
        self.sample_ids = sample_ids
        self.dict_sampleid_to_gates = {}
        
        
class Gating_Interface:
    """
    This class provides a gating GUI.
    
    Parameters
    ----------
    experiment: Experiment
        A CytoPy Experiment object
    source_population: str
        The parent population
    """
    
    def __init__(self,
                 experiment: Experiment,
                 source_population: str
                ):
        
        self.experiment = experiment
        
        #make sure that the source population is valid
        sample_ids = self.experiment.list_samples()
        for sample_id in sample_ids:
            #sample_pops = get_sample_populations(self.experiment, sample_id)
            sample_pops = self.experiment.get_sample(sample_id).list_populations()
            err_msg = 'Population %s not found for sample %s.'%(
                                                     source_population, sample_id)
            err_msg += ' The available populations are: %s'%(sample_pops)
            assert source_population in sample_pops, err_msg
        
        #save gating info
        self.source_population = source_population
        
        #create a FACS_Gater object
        self.gater = FACS_Gater(experiment, parent_population=source_population)
        
        #create layouts and widgets
        self._create_widget_layouts()
        self._create_widgets()

        #mark that the gates were not yet saved
        self.gates_saved = False
        
        #create a first gate
        self._create_gate()
        
        
        
        
    def plot_gates(self):
        assert self.gates_saved, 'No gates were saved yet'
        display(self.gater.fig)        

        
    def _create_widget_layouts(self):    
        """
        define widget layouts
        """
        
        self.widget_small_layout = Layout(width='160px', height='auto')
        self.widget_xsmall_layout = Layout(width='140px', height='auto')
        self.widget_xxsmall_layout = Layout(width='55px', height='auto')
        self.widget_xxsmall2_layout = Layout(width='40px', height='auto')
        self.label_layout = Layout(margin='0px 0px 0px 10px')
        self.label2_layout = Layout(margin='0px 0px 0px 0px')
        self.hbox_layout = Layout(flex_flow='row nowrap',
                                 align_items='stretch',
                                 align_content='stretch',
                                 display='flex',
                                 justify_content='flex-start')
        self.vbox_layout = Layout(flex_flow='column nowrap',
                                 align_items='flex-start',
                                 align_content='flex-start',
                                 display='flex',
                                 justify_content='flex-start')
        
        self.hbox_left_layout = Layout(flex_flow='row nowrap',
                                 align_items='flex-start',
                                 align_content='flex-start',
                                 display='flex',
                                 justify_content='flex-start')
                        
                

    def _create_widgets(self):
        """
        This function creates all the widgets of the gating GUI
        """
        
        #num_preview widget
        self.widget_numpreview = widgets.Dropdown(
                                options=[],
                                description='#preview',
                                layout=self.widget_xsmall_layout
                            )        
        self.widget_numpreview.observe(self._change_numpreview_callback)
        
        #x_min widget
        self.widget_xmin = Text(value='', layout=self.widget_xxsmall_layout)
        self.widget_xmax = Text(value='', layout=self.widget_xxsmall_layout)
        self.widget_ymin = Text(value='', layout=self.widget_xxsmall_layout)
        self.widget_ymax = Text(value='', layout=self.widget_xxsmall_layout)
        self.widget_numpreview.observe(self._change_numpreview_callback)
        

        #prepare top-box
        self.hbox_top = Box([
                            self.widget_numpreview,
                            Label('X range', layout=self.label_layout),
                            self.widget_xmin,
                            Label('-', layout=self.label2_layout),
                            self.widget_xmax,
                            Label('Y range', layout=self.label_layout),
                            self.widget_ymin,
                            Label('-', layout=self.label2_layout),
                            self.widget_ymax,
                            ],
                            layout=self.hbox_layout)
        
        #define x-axis and y-axis widgets
        experiment_channels = get_experiment_channels(self.experiment)
        self.widget_xchannel = widgets.Dropdown(
                                options=experiment_channels,
                                value='FSC-A',
                                description='X channel',
                                layout=self.widget_small_layout
                            )
        self.widget_ychannel = widgets.Dropdown(
                                options=experiment_channels,
                                value='SSC-A',
                                description='Y channel',
                                layout=self.widget_small_layout
                            )
        self.widget_xtransform = widgets.Dropdown(
                                options=['Linear', 'Logicle', 'Log'],
                                description='X transf.',
                                value='Linear',
                                layout=self.widget_small_layout
                            )
        self.widget_ytransform = widgets.Dropdown(
                                options=['Linear', 'Logicle', 'Log'],
                                description='Y transf.',
                                value='Linear',
                                layout=self.widget_small_layout
                            )
        self.hbox_axes = Box([self.widget_xchannel,
                              self.widget_ychannel,
                              self.widget_xtransform,
                              self.widget_ytransform,
                              ],
                              layout=self.hbox_layout)

        #prepare Gate-type selector widget
        self.widget_gatetype = widgets.Dropdown(
                                options=['Ellipse', 'Threshold', 'Threshold1d', 'Clusters'],
                                value='Ellipse',
                                description='Gate Type',
                                layout=self.widget_small_layout
                            )
        self.widget_gatetype.observe(self._change_gate_type_callback)
        
        #prepare create-gate button
        self.widget_createegate = widgets.Button(description="Create gates")
        self.widget_createegate.on_click(self._create_gate_clicked)
        
        #prepare save-gate button
        self.widget_savegate = widgets.Button(description="Save gates", disabled=True)
        self.widget_savegate.on_click(self._save_gate_clicked)

        #prepare boxes that define the layout of all the widgets
        self.widget_output = widgets.Output()
        self.hbox_gate = Box([
                              self.widget_gatetype,
                              self.widget_createegate,
                              self.widget_savegate
                             ],
                             layout=self.hbox_layout)
        self.hbox_gate_params = Box([], layout=self.hbox_left_layout)
        self.vbox = Box([], layout=self.vbox_layout)
        display(self.vbox)
        
        #update the num_preview widget
        sample_ids = self.experiment.list_samples()
        list_numpreview = [c for c in [1,2,4,10] if c <= len(sample_ids)] + ['All']
        self.widget_numpreview.options = list_numpreview
        num_preview = 1
        self.widget_numpreview.value = num_preview
        self._change_numpreview(num_preview)

        #reset the display (like when we change a gate)
        self._change_gate_type(self.widget_gatetype.value)
        
        
    def _change_gate_type(self,
                          gate_type: str
                         ):
        """
        Change the type of gate shown in the GUI
        
        Parameters
        ----------
        gate_type: str
            The gate type
        """
        self.widget_output = widgets.Output()
        self.widget_savegate.disabled = True
        if gate_type == 'Ellipse':
            widgets_list = self._prepare_ellipse_widgets()
        elif gate_type == 'Threshold':
            widgets_list = self._prepare_threshold2d_widgets()
        elif gate_type == 'Threshold1d':
            widgets_list = self._prepare_threshold1d_widgets()
        elif gate_type == 'Clusters':
            widgets_list = self._prepare_clusters_widgets()
        else:
            raise ValueError('unknown gate type: %s'%(gate_type))
            
        self.hbox_gate_params = Box(widgets_list, layout=self.hbox_left_layout)
        self.vbox.children = [self.hbox_top,
                              self.hbox_axes,
                              self.hbox_gate,
                              self.hbox_gate_params,
                              self.widget_output
                             ]
        
        

        
    def _change_numpreview(self,
                           num_preview: int
                          ):
        """
        Change the number of samples that will be previewed
        
        Parameters
        ----------
        num_preview: int
            How many samples will be displayed
        """
        sample_ids = self.experiment.list_samples()
        if num_preview != 'All':
            num_preview = int(num_preview)
            assert len(sample_ids) >= num_preview
            sample_ids = sample_ids[:num_preview]
        self.gater.update_sample_ids(sample_ids)        
        
        
    def _change_numpreview_callback(self, widget):
        if widget['type'] != 'change' or widget['name'] != 'value': return
        self._change_numpreview(widget['new'])

    def _change_gate_type_callback(self, widget):
        if widget['type'] != 'change' or widget['name'] != 'value': return
        self._change_gate_type(widget['new'])
        
        
    def _prepare_ellipse_widgets(self):
        self.widget_popname = Text(
                                   value='A',
                                   description='Target pop',
                                   layout=self.widget_xsmall_layout
                                  )
        self.widget_conf = widgets.Dropdown(
                                options=np.round(np.arange(0.05, 1.01, 0.05), 2),
                                value=0.80,
                                layout=Layout(margin='3px 0px 0px 0px', width='55px'),
                                indent=True
                            )
        self.widget_x = Text(value='', layout=self.widget_xxsmall2_layout, disabled=True, )
        self.widget_y = Text(value='', layout=self.widget_xxsmall2_layout, disabled=True, )
        self.widget_width = Text(value='', layout=self.widget_xxsmall2_layout, disabled=True, )
        self.widget_height = Text(value='', layout=self.widget_xxsmall2_layout, disabled=True, )
        self.widget_angle = Text(value='', layout=self.widget_xxsmall_layout, disabled=True, )
        self.widget_auto_ellipse = Checkbox(value=True,
                                            layout=Layout(
                                                width='15px',
                                                height='auto',
                                                margin='3px 0px 0px 0px'
                                            ),
                                            indent=False)
        self.widget_auto_ellipse.observe(self._toggle_auto_ellipse, 'value')
        return [
                self.widget_popname,
                Label('Auto', layout=self.label_layout), 
                self.widget_auto_ellipse,
                Label('Conf.', layout=Layout(margin='0px 0px 0px 5px')), 
                self.widget_conf,
                Label('X', layout=self.label_layout), self.widget_x,
                Label('Y', layout=self.label_layout), self.widget_y,
                Label('Width', layout=self.label_layout), self.widget_width,
                Label('Height', layout=self.label_layout), self.widget_height,
                Label('Angle', layout=self.label_layout), self.widget_angle,
               ]
    
        
    def _prepare_threshold2d_widgets(self):
        self.widget_x = Text(
                             '200',
                             description='x threshold',
                             disabled=True,
                             layout=self.widget_small_layout
                            )
        self.widget_y = Text(
                             '200',
                             description='y threshold',
                             disabled=True,
                             layout=self.widget_small_layout
                            )
        self.widget_automeans = Checkbox(description='Automatic threshold',
                                         value=True,
                                         indent=False,
                                         layout=self.widget_small_layout)
        self.widget_automeans.observe(self._toggle_auto_means, 'value')
        self.widget_bandwidth = Text('',
                                     description='Bandwidth',
                                     layout=self.widget_xsmall_layout
                                    )
        return [self.widget_automeans, self.widget_x, self.widget_y, self.widget_bandwidth]
    
    def _prepare_threshold1d_widgets(self):
        self.widget_x = Text('200',
                             description='x threshold',
                             disabled=True,
                             layout=self.widget_small_layout
                            )
        self.widget_automeans = Checkbox(description='Automatic threshold',
                                         value=True,
                                         indent=False,
                                         layout=self.widget_small_layout)
        self.widget_automeans.observe(self._toggle_auto_means, 'value')
        self.widget_bandwidth = Text('',
                                     description='Bandwidth',
                                     layout=self.widget_xsmall_layout
                                    )
        return [self.widget_automeans, self.widget_x, self.widget_bandwidth]
        
    
    
    def _prepare_clusters_widgets(self):
        self.widget_nclusters = widgets.Dropdown(
                                options=np.arange(1,9),
                                value=1,
                                description='#clusters',
                                layout=self.widget_small_layout
                                )
        return [self.widget_nclusters]
    

    def _create_gate_clusters(self):
        """
        Create a clustering gate
        """
        self.gater.polygon_gate(gate_name=generate_random_name(),
                                n_clusters=self.widget_nclusters.value)

        
    def _create_gate_ellipse(self):
        """
        Create an ellipse gate
        """
        if len(self.widget_popname.value)==0:
            logging.error('Please provide a target population name')
            return
        
        #determine if it's a manual or automatic ellipse
        if self.widget_auto_ellipse.value:
            centroid, width, height, angle = None, None, None, None
            conf = self.widget_conf.value
            method = 'GaussianMixture'
            
        else:
            conf = None
            method = 'manual'
            
            try:
                x = float(self.widget_x.value)
            except ValueError:
                logging.error('X value is not a number')
                return
            try:
                y = float(self.widget_y.value)
            except ValueError:
                logging.error('Y value is not a number')
                return
            try:
                width = float(self.widget_width.value)
            except ValueError:
                logging.error('Width value is not a number')
                return
            try:
                height = float(self.widget_height.value)
            except ValueError:
                logging.error('Height value is not a number')
                return
            try:
                angle = float(self.widget_angle.value)
            except ValueError:
                logging.error('Angle value is not a number')
                return
            centroid = (x,y)
            
        if len(self.widget_popname.value)==0:
            logging.error('Please provide a target population name')
            return
        

        #create the gates
        self.gater.ellipse_gate(gate_name=generate_random_name(),
                                target_pop_name=self.widget_popname.value,
                                method=method,
                                conf=conf,
                                centroid=centroid,
                                width=width,
                                height=height,
                                angle=angle
                               )
        
        #update ellipse parameters if we only have one ellipse
        if (
            self.widget_numpreview.value == 1
        and self.widget_xtransform.value=='Linear'
        and self.widget_ytransform.value == 'Linear'
        and self.widget_auto_ellipse.value
           ):
            gate = self.gater.gate
            ellipse_means = gate.model.means_
            ellipse_params = [probablistic_ellipse(covar, conf=gate.conf)
                              for covar in gate.model.covariances_]
            assert len(ellipse_means) == 1
            assert len(ellipse_params) == 1
            width = ellipse_params[0][0]
            height = ellipse_params[0][1]
            angle = ellipse_params[0][2]
            x, y = ellipse_means[0][0], ellipse_means[0][1]
            
            self.widget_x.value = str(int(x))
            self.widget_y.value = str(int(y))
            self.widget_height.value = str(int(height))
            self.widget_width.value = str(int(width))
            self.widget_angle.value = str(int(angle))
            
            
            
        
        
    def _create_gate_threshold2d(self):
        """
        Create a threshold2d gate
        """
        
        #settings for automtic thresholds
        if self.widget_automeans.value:
            x_threshold = None
            y_threshold = None
            threshold_method = 'density'
            try:
                if len(self.widget_bandwidth.value)>0:
                        bandwidth = float(self.widget_bandwidth.value)
                else: bandwidth = None
            except ValueError:
                logging.error('bandwidth value is not a number')
                return            
            
        #settings for manual thresholds
        else:
            threshold_method = 'manual'
            bandwidth = None
            try:
                x_threshold = float(self.widget_x.value)
            except ValueError:
                logging.error('x value is not a number')
                return
            try:
                y_threshold = float(self.widget_y.value)
            except ValueError:
                logging.error('y value is not a number')
                return

        #create the gates
        try:
            self.gater.threshold_gate(gate_name=generate_random_name(),
                                      method=threshold_method,
                                      x_threshold=x_threshold,
                                      y_threshold=y_threshold,
                                      bandwidth=bandwidth)
        except ValueError as e:
            if str(e) == 'Unable to solve for support numerically. Use a kernel with finite support or scale data to smaller bw.':
                logging.error('Threhold gate could not use the provided bandwidth \
                               (or find it automatically). \
                               Please provide a bandwidth parameter'
                             )
            else:
                raise
        
        
    def _create_gate_threshold1d(self):
        """
        Create a threshold1d gate
        """
        
        #define a manual threshold if needed
        if self.widget_automeans.value:
            x_threshold = None
            threshold_method = 'density'
        else:
            threshold_method = 'manual'
            try:
                x_threshold = float(self.widget_x.value)
            except ValueError:
                logging.error('x value is not a number')
                return

        #bandwidth is used for both manual and automatic gating
        #(for the purposes of plotting the histogram)
        try:
            if len(self.widget_bandwidth.value)>0:
                bandwidth = float(self.widget_bandwidth.value)
            else:
                bandwidth = None
        except ValueError:
            logging.error('bandwidth value is not a number')
            return

        #create the gates
        try:
            self.gater.threshold_gate(gate_name=generate_random_name(),
                                      method=threshold_method,
                                      x_threshold=x_threshold,
                                      bandwidth=bandwidth
                                     )
        except ValueError as e:
            if str(e) == 'Unable to solve for support numerically. Use a kernel with finite support or scale data to smaller bw.':
                logging.error('Threhold gate could not use the provided bandwidth \
                               (or find it automatically). \
                               Please provide a bandwidth parameter')
            else:
                raise
            
        

    def _create_gate_clicked(self, _):
        self._create_gate()

    def _save_gate_clicked(self, _):
        
        self.gater.save_populations()
        self.gates_saved = True
        with self.widget_output:
            logging.info('Saved populations to DB')
        
        #disable all the widgets
        for hbox in self.vbox.children:
            try: hbox.children
            except AttributeError: continue
            for widget_obj in hbox.children:
                widget_obj.disabled = True
                
        #delete the gating object to save memory
        del self.gater
                
        
    
    def _create_gate(self):
        """
        Create a gate (this is a general function
        that calls a gate-specific function depending on the gate type)
        """
        
        #update gater object
        transform_x = None if self.widget_xtransform.value=='Linear' \
                           else self.widget_xtransform.value.lower() 
        transform_y = None if self.widget_ytransform.value=='Linear' \
                           else self.widget_ytransform.value.lower()
        x = self.widget_xchannel.value
        y = self.widget_ychannel.value
        if self.widget_gatetype.value == 'Threshold1d': y = None
            
        
        #verify that axis limits are valid
        with self.widget_output:
            try:
                xmin = float(self.widget_xmin.value) if self.widget_xmin.value!='' \
                                                     else None
                xmax = float(self.widget_xmax.value) if self.widget_xmax.value!='' \
                                                     else None
            except ValueError:
                logging.error('X range is not a a number')
                return
            try:
                ymin = float(self.widget_ymin.value) if self.widget_ymin.value!='' \
                                                     else None
                ymax = float(self.widget_ymax.value) if self.widget_ymax.value!='' \
                                                     else None
            except ValueError:
                logging.error('Y range is not a a number')
                return
        
        
        self.gater.update_params(
            transform_x=transform_x,
            transform_y=transform_y,
            x=x,
            y=y,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax
        )
        
        with self.widget_output:
            clear_output(wait=True)
            gate_type = self.widget_gatetype.value
            if gate_type == 'Ellipse':
                self._create_gate_ellipse()
            elif gate_type == 'Threshold':
                self._create_gate_threshold2d()
            elif gate_type == 'Threshold1d':
                self._create_gate_threshold1d()
            elif gate_type == 'Clusters':
                self._create_gate_clusters()
            else:
                raise ValueError('unknown gate type: %s'%(gate_type))
            show_inline_matplotlib_plots()
            
        #enable gate saving if we view all gates
        if self.widget_numpreview.value == 'All':
            self.widget_savegate.disabled = False

        
    def _toggle_auto_means(self, checkbox):
        """
        Disable or enable the x and y text boxes, depending on the value in the checkbox
        """
        self.widget_x.disabled = checkbox['new']
        self.widget_y.disabled = checkbox['new']

        
    def _toggle_auto_ellipse(self, checkbox):
        """
        Disable or enable the manual ellipse widgets
        """
        self.widget_x.disabled = checkbox['new']
        self.widget_y.disabled = checkbox['new']
        self.widget_width.disabled = checkbox['new']
        self.widget_height.disabled = checkbox['new']
        self.widget_angle.disabled = checkbox['new']
        self.widget_conf.disabled = not checkbox['new']
        

        
        