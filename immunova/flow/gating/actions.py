from immunova.data.gating import Gate, GatingStrategy
from immunova.data.fcs import FileGroup, Population
from immunova.data.fcs_experiments import FCSExperiment
from immunova.flow.gating.static import rect_gate
from immunova.flow.gating.fmo import density_2d_fmo, density_1d_fmo
from immunova.flow.gating.density import density_gate_1d
from immunova.flow.gating.mixturemodel import mm_gate, inside_ellipse
from immunova.flow.gating.dbscan import dbscan_gate
from immunova.flow.gating.quantile import quantile_gate
from immunova.flow.gating.utilities import apply_transform
from immunova.flow.gating.defaults import Geom
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import pandas as pd
import inspect


class Gating:

    def __init__(self, experiment: FCSExperiment, sample_id: str, transformation: str or None = "logicle",
                 transform_channels: list or None = None, data_type='raw', sample: int or None = None):
        data, mappings = experiment.pull_sample_data(sample_id=sample_id, data_type=data_type, sample_size=sample)
        primary = [x for x in data if x['typ'] == 'complete'][0]
        controls = [x for x in data if x['typ'] == 'control']
        fg = FileGroup.objects(primary_id=sample_id)[0]
        self.mappings = mappings
        self.id = sample_id
        self.experiment_id = FCSExperiment.experiment_id

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
            root = dict(population_name='root', prop_of_parent=1.0, prop_of_total=1.0,
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
        # ToDo use CV grid search to optimise parameters
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

    def apply(self, gate_name: str, add_population: bool = False, plot_output: bool = True):
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
        fmo_x_name, fmo_y_name = None, None
        if 'fmo_x' in kwargs.keys():
            fmo_x_name = kwargs['fmo_x']
            kwargs['fmo_x'] = self.knn_fmo(gate.parent, fmo_x_name)
        if 'fmo_y' in kwargs.keys():
            fmo_y_name = kwargs['fmo_y']
            kwargs['fmo_y'] = self.knn_fmo(gate.parent, fmo_y_name)

        parent_population = self.get_population_df(gate.parent)
        output = func(data=parent_population, **kwargs)
        if output.error:
            print(output.error_msg)
            return None
        if add_population:
            for name, data in output.child_populations.items():
                # Check gate type corresponds to output
                if gate.gate_type == 'geom' and data['geom'] is None:
                    print(f'Error: Geom gate returning null value for child population ({name}) geom')
                    return None
                n = len(data['index'])
                self.populations[name] = dict(population_name=name, index=data['index'],
                                              prop_of_parent=n/parent_population.shape[0],
                                              prop_of_total=n/self.data.shape[0],
                                              parent=gate.parent, children=[],
                                              geom=data['geom'])
                self.populations[gate.parent]['children'].append(name)
        if plot_output:
            self.plot_gate(gate.gate_name)
            if fmo_x_name:
                fmo = kwargs['fmo_x']
                self.plot_gate(gate.gate_name, fmo=fmo, title=fmo_x_name)
            if fmo_y_name:
                fmo = kwargs['fmo_y']
                self.plot_gate(gate.gate_name, fmo=fmo, title=fmo_y_name)
        return output

    def apply_many(self, gates: list = None, apply_all=False, plot_outcome=False):
        gating_results = dict()
        if apply_all:
            if len(self.populations.keys()) != 1:
                print('User has chosen to apply all gates on a file with existing populations, '
                      'when using the `apply_all` command files should have no existing populations. '
                      'Remove existing populations from file before continuing. Aborting.')
                return None
            gates_to_apply = self.gates
        else:
            if any([x not in self.gates.keys() for x in gates]):
                print(f'Error: some gate names provided appear invalid; valid gates: {self.gates.keys()}')
                return None
            gates_to_apply = [name for name, _ in self.gates.items() if name in gates]
        for gate_name in gates_to_apply:
            gating_results[gate_name] = self.apply(gate_name, add_population=True, plot_output=plot_outcome)
        return gating_results

    @staticmethod
    def __plot_axis_lims(x, y, xlim=None, ylim=None):
        if not xlim:
            if any([x.find(c) != -1 for c in ['FSC', 'SSC']]):
                xlim = (0, 250000)
            else:
                xlim = (0, 1)
        if not ylim:
            if any([y.find(c) != -1 for c in ['FSC', 'SSC']]):
                ylim = (0, 250000)
            else:
                ylim = (0, 1)
        return xlim, ylim

    def plot_gate(self, gate_name, title=None, fmo: str or pd.DataFrame or None = None,
                  xlim=None, ylim=None, show=True):
        # Check and load gate
        if gate_name not in self.gates.keys():
            print(f'Error: invalid gate name, must be one of {self.gates.keys()}')
            return None
        gate = self.gates[gate_name]

        # Get axis info
        x = gate.x_feature
        if gate.y_feature:
            y = gate.y_feature
        else:
            y = 'FSC-A'
        xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)

        # Cluster plot
        if gate.gate_type == 'cluster':
            return self.__cluster_plot(x, y, gate)

        # Geom plot
        if type(fmo) == str:
            data = self.knn_fmo(gate.parent, fmo)
        elif type(fmo) == pd.DataFrame:
            data = fmo
            del fmo
        elif fmo is None:
            data = self.get_population_df(gate.parent)
        else:
            print('Error: invalid input type for fmo, must be either string, pandas dataframe or None')
            return None
        plots = []
        for child in self.populations[gate.parent]['children']:
            plots.append(self.__geom_plot(x, y, data, self.populations[gate.parent]['geom'],
                                          xlim, ylim))
        return plots

    def __cluster_plot(self, x, y, gate):
        fig, ax = plt.subplots(figsize=(5, 5))
        colours = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(gate.children))]
        for child, colour in zip(gate.children, colours):
            d = self.get_population_df(child).sample(frac=0.25)
            if child == 'noise':
                colour = [0, 0, 0, 1]
            ax.scatter(d[x], d[y], c=colour, s=1, alpha=0.25)
        fig.show()

    @staticmethod
    def __standard_2dhist(ax, data, x, y, xlim, ylim):
        ax.hist2d(data[x], data[y], bins=500, norm=LogNorm())
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(y)
        ax.set_xlabel(x)
        return ax

    def __geom_plot(self, x, y, data, geom, xlim, ylim):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax = self.__standard_2dhist(ax, data, x, y, xlim, ylim)
        if 'threshold' in geom.keys():
            ax.axvline(geom['threshold'], c='r')
        if 'threshold_x' in geom.keys():
            ax.axvline(geom['threshold_x'], c='r')
        if 'threshold_y' in geom.keys():
            ax.axhline(geom['threshold_y'], c='r')
        if all([x in geom.keys() for x in ['mean', 'width', 'height', 'angle']]):
            ellipse = patches.Ellipse(xy=geom['mean'], width=geom['width'], height=geom['height'],
                                      angle=geom['angle'], fill=False, edgecolor='r')
            ax.add_patch(ellipse)
        if all([x in geom.keys() for x in ['x_min', 'x_max', 'y_min', 'y_max']]):
            rect = patches.Rectangle(xy=(geom['x_min'], geom['y_min']),
                                     width=geom['x_max'], height=geom['y_max'],
                                     fill=False, edgecolor='r')
            ax.add_patch(rect)
        fig.show()
        return fig

    def plot_populations(self, population_name, x, y, xlim=None, ylim=None, show=True):
        fig, ax = plt.subplots(figsize=(5, 5))
        if population_name in self.populations.keys():
            p = self.populations[population_name]
        else:
            print(f'Invalid population name, must be one of {self.populations.keys()}')
            return None
        xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
        data = self.get_population_df(p)
        self.__standard_2dhist(ax, data, x, y, xlim, ylim)
        if show:
            fig.show()
        return fig

    def find_dependencies(self, population: str = None) -> list or None:
        """
        For a given population find all dependencies
        :param population: population name
        :return: list of Gate objects dependent on given gate/population
        """
        if population not in self.populations.keys():
            print(f'Population {population} does not exist')
            return []
        dependencies = self.populations[population]['children']
        dependencies = [p for p in dependencies if p != population]
        for child in dependencies:
            dependencies = dependencies + self.find_dependencies(child)
        return dependencies

    def remove_population(self, population_name: str):
        if population_name not in self.populations.keys():
            print(f'{population_name} does not exist')
            return None
        downstream_populations = self.find_dependencies(population=population_name)
        removed = []
        # Remove populations downstream
        if downstream_populations:
            for p in downstream_populations:
                removed.append(self.populations.pop(p))
        # Updated children in parent
        parent = self.populations[population_name]['parent']
        self.populations[parent]['children'] = [x for x in self.populations[parent]['children']
                                                if x != population_name]
        removed.append(self.populations.pop(population_name))
        return removed

    def remove_gate(self, gate_name, propagate=True):
        if gate_name not in self.gates.keys():
            print('Error: invalid gate name')
            return None
        gate = self.gates[gate_name]
        if not gate.children or not propagate:
            self.gates.pop(gate_name)
            return True
        # Remove affected gates and downstream populations
        effected_populations = []
        for child in gate.children:
            effected_populations = effected_populations + self.find_dependencies(population=child)
            self.remove_population(child)
            effected_populations.append(child)
        effected_gates = [name for name, gate in self.gates.items() if gate.parent in effected_populations]
        effected_gates.append(gate_name)
        for g in effected_gates:
            self.gates.pop(g)
        return effected_populations, effected_gates

    @staticmethod
    def __update_population(updated_geom, parent_population, bool_gate):
        try:
            new_population = None
            if updated_geom['shape'] == 'threshold':
                x = updated_geom['x']
                new_population = parent_population[parent_population[x] >= updated_geom['threshold']]
            if updated_geom['shape'] == 'rect':
                x = updated_geom['x']
                y = updated_geom['y']
                new_population = parent_population[(parent_population[x] >= updated_geom['x_min']) &
                                                   (parent_population[x] < updated_geom['x_max'])]
                new_population = new_population[(new_population[y] >= updated_geom['y_min']) &
                                                (new_population[y] < updated_geom['y_max'])]
            if updated_geom['shape'] == 'ellipse':
                data = parent_population[[updated_geom['x'], updated_geom['y']]].values
                new_population = inside_ellipse(data, width=updated_geom['width'],
                                                height=updated_geom['height'],
                                                angle=updated_geom['angle'],
                                                center=updated_geom['mean'])
                new_population = parent_population[new_population]
            if new_population is None:
                print('Error: Geom shape is not recognised, expecting one of: threshold, 2d_threshold, '
                      'rect, or ellipse')
                return None
            if bool_gate:
                new_population = parent_population[~parent_population.index.isin(new_population.index)]
            return new_population
        except KeyError as e:
            print(f'Error, invalid Geom: {e}')
            return None

    def update_geom(self, population_name, updated_geom, bool_gate=False):
        if population_name not in self.populations.keys():
            print(f'Error: population name {population_name} not recognised')
            return None
        parent_population = self.get_population_df(self.populations[population_name]['parent'])
        # Generate new population with new definition
        new_population = self.__update_population(updated_geom, parent_population, bool_gate)
        if new_population is None:
            return None
        # Calculate downstream effects
        dependent_pops = self.find_dependencies(population=population_name)
        dependent_pops.append(population_name)
        parent_name = self.populations[population_name]['parent']
        self.remove_population(population_name)
        n = new_population.shape[0]
        self.populations[population_name] = dict(population_name=population_name,
                                                 index=new_population.index.values,
                                                 children=[], parent=parent_name,
                                                 prop_of_parent=n/parent_population.shape[0],
                                                 prop_of_total=n/self.data.shape[0],
                                                 geom=updated_geom, warnings=['Manual gate'])
        effected_gates = [name for name, gate in self.gates.items() if gate.parent in dependent_pops]
        if effected_gates:
            print(f'Recomputing: {effected_gates}')
            return self.apply_many(gates=effected_gates)
        print('No downstream effects, re-gating complete!')
        return None

    def __population_to_mongo(self, population_name):
        pop_mongo = Population()
        pop_dict = self.populations[population_name]
        for k in pop_dict.keys():
            if k != 'index':
                pop_mongo[k] = pop_dict[k]
            else:
                pop_mongo.save_index(pop_dict[k])
        return pop_mongo

    def save(self, overwrite=False):
        fg = FileGroup.objects(primary_id=self.id)[0]
        existing_cache = fg.populations
        existing_gates = fg.gates
        if existing_cache or existing_gates:
            if not overwrite:
                print(f'Population cache for fcs file {self.id} already exists, if you wish to overwrite'
                      f'this cache then specify overwrite as True')
                return False
            fg.populations = []
            fg.gates = []
            fg.save()
        for name in self.populations.keys():
            FileGroup.objects(primary_id=self.id).update(push__populations=self.__population_to_mongo(name))
        for _, gate in self.gates.items():
            FileGroup.objects(primary_id=self.id).update(push__gates=gate)
        print('Saved successfully!')
        return True


class Template(Gating):
    def save_new_template(self, template_name, overwrite=True):
        gating_strategy = GatingStrategy.objects(template_name=template_name)
        if gating_strategy:
            if not overwrite:
                print(f'Template with name {template_name} already exists, set parameter '
                      f'`overwrite` to True to continue')
                return False
            print(f'Overwriting existing gating template {template_name}')
            gating_strategy = gating_strategy[0]
            gating_strategy.gates = self.gates
            gating_strategy.last_edit = datetime.now()
            gating_strategy.save()
            exp = FCSExperiment.objects(experiment_id=self.experiment_id)
            templates = [x for x in exp.gating_templates if x.template_name != gating_strategy.template_name]
            templates.append(gating_strategy)
            exp.gating_templates = templates
            exp.save()
            return True
        else:
            print(f'No existing template named {template_name}, creating new template')
            gating_strategy = GatingStrategy()
            gating_strategy.template_name = template_name
            gating_strategy.creation_date = datetime.now()
            gating_strategy.last_edit = datetime.now()
            gating_strategy.gates = self.gates
            gating_strategy.save()
            FCSExperiment.objects(experiment_id=self.experiment_id).update(push__gating_templates=gating_strategy)
            return True

    def load_template(self, template_name):
        gating_strategy = GatingStrategy.objects(template_name=template_name)
        if gating_strategy:
            self.gates = gating_strategy[0].gates
            return True
        else:
            print(f'No template with name {template_name}')
            return False





