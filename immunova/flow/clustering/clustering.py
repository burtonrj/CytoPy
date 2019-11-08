from immunova.flow.gating.actions import Gating


class ClusteringError(Exception):
    pass


class Clustering:
    def __init__(self, experiment, sample_id, root_population, n_jobs=-1, transform: bool = True,
                 transform_method: str = 'logicle', transform_features: str or list = 'all'):
        self.experiment = experiment
        self.n_jobs = n_jobs
        self.cluster_cache = dict()
        g = Gating(experiment, sample_id)
        self.data = g.get_population_df(root_population,
                                        transform=transform,
                                        transform_method=transform_method,
                                        transform_features=transform_features)
        if self.data is None:
            raise ClusteringError(f'Error: was unable to retrieve data for {sample_id} invalid root population, '
                                  f'{g.populations.keys()}')

    def save(self):
        pass

    def load(self):
        pass
