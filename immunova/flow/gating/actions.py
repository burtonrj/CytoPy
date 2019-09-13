from data.gating import Gate, GatingStrategy
from data.fcs_experiments import FCSExperiment
from data.fcs import FileGroup, Population
from data.fcs_experiments import FCSExperiment
from flow.gating.static import rect_gate
from flow.gating.fmo import density_2d_fmo, density_1d_fmo
from flow.gating.density import density_gate_1d
from flow.gating.mixturemodel import mm_gate, inside_ellipse
from flow.gating.dbscan import dbscan_gate
from flow.gating.quantile import quantile_gate
from flow.gating.utilities import apply_transform
from flow.gating.defaults import GateOutput, Geom
from datetime import datetime
from data.fcs import FileGroup
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import inspect
import functools


class Gating:

    def __init__(self, experiment: FCSExperiment, sample_id: str, transformation: str or None = "logicle",
                 transform_channels: list or None = None, data_type='raw', sample: int or None = None):
        data, mappings = experiment.pull_sample_data(sample_id=sample_id, data_type=data_type, sample_size=sample)
        primary = [x for x in data if x['typ'] == 'complete'][0]
        controls = [x for x in data if x['typ'] == 'control']
        fg = FileGroup.objects(primary_id=sample_id)[0]
        self.mappings = mappings
        self.id = sample_id

        if transformation:
            if not transform_channels:
                transform_channels = [x for x in primary['data'].columns if
                                      all([x.find(y) == -1 for y in ['FSC', 'SSC', 'Time']])]
            self.data = apply_transform(primary['data'].copy(), features_to_transform=transform_channels,
                                        transform_method=transformation)
            self.fmo = {x['id']: apply_transform(x['data'], features_to_transform=transform_channels,
                                                 transform_method=transformation) for x in controls}
        else:
            self.data = primary['data']
            self.fmo = {x['id']: x['data'] for x in controls}
        del data

        self.gates = dict()
        if fg.gates:
            for g in fg.gates:
                self.gates[g.gate_name] = g

        self.populations = dict()
        if fg.populations:
            for p in fg.populations:
                self.populations[p.population_name] = p.to_python()
        else:
            root = Population(population_name='root', prop_of_parent=1.0, prop_of_total=1.0,
                              warnings=[], parent='NA', children=[], geom=Geom(shape=None, x='FSC-A', y='SSC-A'),
                              index=self.data.index.values)
            self.populations['root'] = root

    @property
    def gating_functions(self):
        available_functions = [rect_gate, density_gate_1d, density_1d_fmo, density_2d_fmo,
                               mm_gate, dbscan_gate, quantile_gate]
        return {x.__name__: x for x in available_functions}

    def get_population_df(self, population_name: str) -> pd.DataFrame or None:
        """
        Retrieve a population as a pandas dataframe
        :param population_name: name of population to retrieve
        :return: Population dataframe
        """
        if population_name not in self.populations.keys():
            print(f'Population {population_name} not recognised')
            return None
        idx = self.populations[population_name]['index']
        return self.data[self.data.index.isin(idx)]

    def knn_fmo(self, population_to_excerpt, fmo):
        """
        Using the gated population in the whole panel, create a nearest neighbours model
        to predict the labels of FMO data
        :param population_to_excerpt:
        :param fmo:
        :return:
        """
        # Get Data
        if f'{self.id}_{fmo}' != fmo:
            fmo = f'{self.id}_{fmo}'
        parent = self.populations[population_to_excerpt]['parent']
        xf = self.populations[population_to_excerpt]['geom'].x
        yf = self.populations[population_to_excerpt]['geom'].y
        parent_data = self.get_population_df(parent)
        parent_data['pos'] = 0
        # Label data
        true_labels = self.get_population_df(population_to_excerpt).index.values
        for i in true_labels:
            parent_data.set_value(i, 'pos', 1)
        X = parent_data[[xf, yf]]
        y = parent_data['pos']
        # Resample for class imbalance
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        knn = KNeighborsClassifier()
        knn.fit(X_resampled, y_resampled)
        fmo_data = self.fmo[fmo]
        x = knn.predict(fmo_data[[xf, yf]])
        fmo_data['pos'] = x
        return fmo_data[fmo_data['pos'] == 1]

    @staticmethod
    def __check_func_args(func, **kwargs):
        expected_args = [k for k, v in inspect.signature(func).parameters.items()
                         if v.default is inspect.Parameter.empty]
        for arg in expected_args:
            if arg == 'data':
                continue
            if arg not in kwargs.keys():
                print(f'Error: missing required argument {arg} for gating function {func.__name__}')
                return False
        return True

    def create_gate(self, gate_name, children, parent, x, y, func, func_args, gate_type, boolean_gate=False):
        if gate_name in self.gates.keys():
            print(f'Error: gate with name {gate_name} already exists.')
            return False
        if func not in self.gating_functions:
            print(f'Error: invalid gate function, must be one of {self.gating_functions}')
        if not self.__check_func_args(self.gating_functions[func], **func_args):
            return False
        func_args = [(k, v) for k, v in func_args.items()]
        new_gate = Gate(gate_name=gate_name, children=children, parent=parent,
                        x=x, y=y, func=func, func_args=func_args, gate_type=gate_type,
                        boolean_gate=boolean_gate)
        self.gates[gate_name] = new_gate
        return True

    def apply(self, gate_name: str, save: bool = False, plot_output: bool = True):
        if gate_name not in self.gates.keys():
            print(f'Error: {gate_name} does not exist. You must create this gate first using the create_gate method')
            return None
        gate = self.gates[gate_name]
        if gate.parent not in self.populations.keys():
            print('Invalid parent; does not exist in current Gating object')
            return None
        for c in gate.children:
            if c in self.populations.keys():
                print(f'Error: population {c} already exists, if you wish to overwrite this population please remove'
                      f' it with the remove_population method and then try again')
                return None
        func = self.gating_functions[gate.func]
        kwargs = {k: v for k, v in gate.func_args}
        if 'fmo_x' in kwargs.keys():
            kwargs['fmo_x'] = self.knn_fmo(gate.parent, kwargs['fmo_x'])
        if 'fmo_y' in kwargs.keys():
            kwargs['fmo_y'] = self.knn_fmo(gate.parent, kwargs['fmo_y'])

        parent_population = self.get_population_df(gate.parent)
        output = func(data=parent_population, **kwargs)
        if output.error:
            print(output.error_msg)
            return None
        if save:
            # ToDo save population
            pass
        if plot_output:
            # ToDo call function for plotting
            pass
        return output

