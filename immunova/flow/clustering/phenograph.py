import phenograph
from immunova.flow.gating.actions import Gating
import umap
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import PreText, Select
from bokeh.plotting import figure


class PhenoGraph:
    def __init__(self, exp, k=30, n_jobs=-1):
        self.experiment = exp
        self.k = k
        self.n_jobs=n_jobs
        self.cluster_cache = dict()

    def cluster_sample(self, sample_id, features, root_population):
        if sample_id in self.cluster_cache.keys():
            return self.cluster_cache[sample_id]
        if sample_id not in self.experiment.list_samples():
            print(f'Invalid sample ID, must be one of: {self.experiment.list_samples()}')
            return None
        data = Gating(self.experiment, sample_id).get_population_df(root_population)
        if not all([x in data.columns for x in features]):
            print(f'Invalid Features, must be one of: {data.columns}')
            return None
        communities, graph, Q = phenograph.cluster(data[features])
        self.cluster_cache[sample_id] = dict(communities=communities, graph=graph, Q=Q)
        return self.cluster_cache[sample_id]
