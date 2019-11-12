from immunova.flow.clustering.plotting.static import label_dataset_clusters, dimensionality_reduction, sample_data

from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LinearColorMapper, BasicTicker, ColorBar
from bokeh.transform import factor_cmap
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel
import colorcet as cc

import pandas as pd
from math import pi


def umap_heatmap(data, clusters, title, output_path):
    output_file(f'{output_path}/{title}.html')

    # Pre-process data
    data = label_dataset_clusters(data, clusters)
    sample = sample_data(data, n=50000, method='uniform')
    features = [x for x in data.columns if x != 'clusters']
    umap_data = dimensionality_reduction(sample, features, method='umap', n_components=2)

    hm_data = sample.groupby(by='clusters').mean()
    hm_data.columns.name = 'channel'
    df = pd.DataFrame(hm_data.stack(), columns=['MFI']).reset_index()
    data = ColumnDataSource(data=dict(umap_x=umap_data['umap0'],
                                      umap_y=umap_data['umap1'],
                                      umap_fill=umap_data['clusters'],
                                      hm_y=hm_data['clusters'],
                                      hm_x=hm_data['channel'],
                                      hm_fill=hm_data['MFI']))

    # Heatmap
    hm_colours = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2",
                  "#dfccce", "#ddb7b1", "#cc7878", "#933b41",
                  "#550b1d"]
    tools = "hover,save,reset"
    mapper = LinearColorMapper(palette=hm_colours, low=df.MFI.min(), high=df.MFI.max())
    p = figure(title='Mean Fluorescence Intensity per Cluster',
               x_range=list(hm_data.columns), y_range=list(hm_data.index),
               x_axis_location="above", plot_width=600, plot_height=400,
               tools=tools, toolbar_location='below',
               tooltips=[('Cluster', '@hm_y'), ('Channel', '@hm_x'), ('MFI', '@hm_fill')])

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3

    p.rect(x="hm_x", y="hm_y", width=1, height=1,
           source=data,
           fill_color={'field': 'hm_fill', 'transform': mapper},
           line_color=None)
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="10pt",
                         ticker=BasicTicker(desired_num_ticks=len(hm_colours)),
                         label_standoff=6, border_line_color=None, location=(0, 0))
    p.add_layout(color_bar, 'right')

    # UMAP scatter plot
    umap_colours = [cc.rainbow[i*15] for i in range(len(umap_data['clusters'].unique()))]
    q = figure(tools=tools, plot_width=500, plot_height=500, toolbar_location='below',
               tooltips=[('Cluster', '@umap_fill')])
    q.circle(x='umap_x', y='umap_y', source=data, fill_alpha=0.4, line_alpha=0.4, size=1,
             fill_color=factor_cmap('umap_fill', palette=umap_colours, factors=list(umap_data['clusters'].unique())))

    grid = gridplot([[q, p]])
    show(grid)
