# Dependencies
# Immunova.data
from immunova.data.gating import Gate as DataGate, GatingStrategy
from immunova.data.fcs import FileGroup, Population
from immunova.data.fcs_experiments import FCSExperiment
# Immunova.flow
from immunova.flow.gating.base import Gate
from immunova.flow.gating.static import Static
from immunova.flow.gating.fmo import FMOGate
from immunova.flow.gating.density import DensityThreshold
from immunova.flow.gating.dbscan import DensityBasedClustering
from immunova.flow.gating.quantile import Quantile
from immunova.flow.gating.mixturemodel import MixtureModel
from immunova.flow.gating.utilities import inside_ellipse
from immunova.flow.gating.defaults import ChildPopulationCollection
# Housekeeping and other tools
from anytree import Node, RenderTree
from datetime import datetime
from itertools import cycle
import inspect
# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches
# Scipy
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np


class Gating:
    def __init__(self, experiment: FCSExperiment, sample_id: str,
                 data_type='raw', sample: int or None = None):
        try:
            data = experiment.pull_sample_data(sample_id=sample_id, data_type=data_type, sample_size=sample)
            assert data is not None
            self.data = [x for x in data if x['typ'] == 'complete'][0]
            self.fmo = [x for x in data if x['typ'] == 'control']
            fg = FileGroup.objects(primary_id=sample_id)[0]
            self.id = sample_id
            self.experiment_id = experiment.experiment_id
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
                            warnings=[], geom=dict(shape=None, x='FSC-A', y='SSC-A'), index=self.data.index.values)
                self.populations['root'] = root

            self.population_tree = dict()
            for node in fg.population_tree:
                self.population_tree[node.name] = Node()
        except AssertionError:
            print('Error: failed to construct Gating object')

    @property
    def gating_classes(self):
        available_classes = [Static, FMOGate, DensityBasedClustering, DensityThreshold, Quantile, MixtureModel]
        return {x.__name__: x for x in available_classes}

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
        return self.data.loc[idx]

    # ToDO knn_fmo
    def knn_fmo(self):
        pass

    @staticmethod
    def __check_class_args(klass, method, **kwargs):
        try:
            klass_args = [k for k, v in inspect.signature(klass).parameters.items()
                          if v.default is inspect.Parameter.empty]
            for arg in klass_args:
                if arg in ['data']:
                    continue
                if arg not in kwargs.keys():
                    print(f'Error: missing required class constructor argument {arg} '
                          f'for gating class {klass.__name__}')
                    return False
            method_args = [k for k, v in inspect.signature(getattr(klass, method)).parameters.items()
                           if v.default is inspect.Parameter.empty]
            for arg in method_args:
                if arg not in kwargs.keys():
                    print(f'Error: missing required method argument {arg} for method '
                          f'{method} belonging to {klass.__name__}')
                    return False
            return True
        except AttributeError:
            print(f'Error: {method} is not a valid method for class {klass.__name__}')
            return False

    def create_gate(self, gate_name: str, parent: str, class_: str, method: str, kwargs: dict,
                    child_populations: ChildPopulationCollection):
        if gate_name in self.gates.keys():
            print(f'Error: gate with name {gate_name} already exists.')
            return False
        if class_ not in self.gating_classes:
            print(f'Error: invalid gate class, must be one of {self.gating_classes}')
            return False
        kwargs['child_populations'] = child_populations
        if not self.__check_class_args(self.gating_classes[class_], method, **kwargs):
            return False
        kwargs = [(k, v) for k, v in kwargs.items()]
        new_gate = DataGate(gate_name=gate_name, children=list(child_populations.populations.keys()), parent=parent,
                            method=method, kwargs=kwargs, class_=class_)
        self.gates[gate_name] = new_gate
        return True

    def __apply_checks(self, gate_name):
        if gate_name not in self.gates.keys():
            print(f'Error: {gate_name} does not exist. You must create this gate first using the create_gate method')
            return None
        gatedoc = self.gates[gate_name]
        if gatedoc.parent not in self.populations.keys():
            print('Invalid parent; does not exist in current Gating object')
            return None
        for c in gatedoc.children:
            if c in self.populations.keys():
                print(f'Error: population {c} already exists, if you wish to overwrite this population please remove'
                      f' it with the remove_population method and then try again')
                return None
        return gatedoc

    def __construct_class_and_gate(self, gatedoc, kwargs):
        klass = self.gating_classes[gatedoc.class_]
        parent_population = self.get_population_df(gatedoc.parent)
        constructor_args = {k: v for k, v in kwargs if k in inspect.signature(klass).parameters.keys()}
        method_args = {k: v for k, v in kwargs if k in inspect.signature(getattr(klass, gatedoc.method)).parameters.keys()}
        analyst = klass(data=parent_population, **constructor_args)
        output = getattr(analyst, gatedoc.method)(**method_args)
        return self.__update_populations(output, parent=parent_population, warnings=analyst.warnings)

    def apply(self, gate_name: str, plot_output: bool = True):
        gatedoc = self.__apply_checks(gate_name)
        if gatedoc is None:
            return None
        kwargs = {k: v for k, v in gatedoc.kwargs}
        if not any([x in ['fmo_x', 'fmo_y'] for x in kwargs.keys()]):
            print('FMO gating requires that you specify an FMO')
            return None
        if 'fmo_x' in kwargs.keys():
            kwargs['fmo_x'] = self.knn_fmo(kwargs['fmo_x'], gatedoc.parent)
        if 'fmo_y' in kwargs.keys():
            kwargs['fmo_y'] = self.knn_fmo(kwargs['fmo_y'], gatedoc.parent)
        output = self.__construct_class_and_gate(gatedoc, kwargs)
        # ToDo call plotting class

    def __update_populations(self, output: ChildPopulationCollection, parent: pd.DataFrame, warnings: list):

        for name, population in output.populations.items():
            n = len(population.index)
            if n == 0:
                prop_of_total = 0
                prop_of_parent = 0
            else:
                prop_of_parent = n / parent.shape[0]
                prop_of_total = n / self.data.shape[0]
            geom = None
            if population.geom is not None:
                geom = population.geom.as_dict()
            self.populations[name] = dict(population_name=name, index=population.index,
                                          prop_of_parent=prop_of_parent,
                                          prop_of_total=prop_of_total,
                                          geom=geom, warnings=warnings)

            # ToDo Update Tree
        return output

    def apply_many(self, gates: list = None, apply_all=False, plot_outcome=False, feedback=True):
        gating_results = dict()
        if apply_all:
            if len(self.populations.keys()) != 1:
                print('User has chosen to apply all gates on a file with existing populations, '
                      'when using the `apply_all` command files should have no existing populations. '
                      'Remove existing populations from file before continuing. Aborting.')
                return None
            gates_to_apply = self.gates.keys()
        else:
            if any([x not in self.gates.keys() for x in gates]):
                print(f'Error: some gate names provided appear invalid; valid gates: {self.gates.keys()}')
                return None
            gates_to_apply = [name for name, _ in self.gates.items() if name in gates]
        for gate_name in gates_to_apply:
            if feedback:
                print(f'Applying {gate_name}...')
            gating_results[gate_name] = self.apply(gate_name, plot_output=plot_outcome)
        if feedback:
            print('Complete!')
        return gating_results

    def plot_gate(self, gate_name, xlim=None, ylim=None):
        gate = self.gates[gate_name]
        data = dict(primary=self.get_population_df(gate.parent))
        kwargs = {k: v for k, v in gate.func_args}

        def get_fmo_data(fk):
            if fk in kwargs.keys():
                return self.knn_fmo(gate.parent, kwargs[fk])
            return None

        for x in ['fmo_x', 'fmo_y']:
            d = get_fmo_data(x)
            if d is not None:
                data[x] = d

        n = len(data.keys())
        # Get axis info
        x = gate.x
        if gate.y:
            y = gate.y
        else:
            y = 'FSC-A'
        xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
        if gate.gate_type == 'cluster':
            return self.__cluster_plot(x, y, gate, title=gate.gate_name)
        fig, axes = plt.subplots(ncols=n)
        if n > 1:
            for ax, (name, d) in zip(axes, data.items()):
                self.__geom_plot(ax=ax, x=x, y=y, data=d, geom=self.populations[gate.children[0]]['geom'],
                                 xlim=xlim, ylim=ylim, title=f'{gate.gate_name}_{name}')
        else:
            self.__geom_plot(ax=axes, x=x, y=y, data=data['primary'], geom=self.populations[gate.children[0]]['geom'],
                             xlim=xlim, ylim=ylim, title=gate.gate_name)
        return fig, axes

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

    def __cluster_plot(self, x, y, gate, title):
        fig, ax = plt.subplots(figsize=(5, 5))
        colours = cycle(['black', 'green', 'blue', 'red', 'magenta', 'cyan'])
        for child, colour in zip(gate.children, colours):
            d = self.get_population_df(child)
            if d is not None:
                d.sample(frac=0.5)
            else:
                continue
            ax.scatter(d[x], d[y], c=colour, s=3, alpha=0.4)
        ax.set_title(title)
        fig.show()

    def __standard_2dhist(self, ax, data, x, y, xlim, ylim, title):
        if data.shape[0] <= 100:
            bins = 50
        elif data.shape[0] > 1000:
            bins = 500
        else:
            bins = int(data.shape[0]*0.5)
        ax.hist2d(data[x], data[y], bins=bins, norm=LogNorm())
        ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)
        return ax

    @staticmethod
    def __plot_asthetics(ax, x, y, xlim, ylim, title):
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_ylabel(y)
        ax.set_xlabel(x)
        ax.set_title(title)
        return ax

    def __geom_plot(self, ax, x, y, data, geom, xlim, ylim, title):
        if data.shape[0] > 1000:
            ax = self.__standard_2dhist(ax, data, x, y, xlim, ylim, title)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)
        else:
            ax.scatter(x=data[x], y=data[y], s=3)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title)
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
                                     width=((geom['x_max']) - (geom['x_min'])),
                                     height=(geom['y_max'] - geom['y_min']),
                                     fill=False, edgecolor='r')
            ax.add_patch(rect)
        return ax

    def plot_population(self, population_name, x, y, xlim=None, ylim=None, show=True):
        fig, ax = plt.subplots(figsize=(5, 5))
        if population_name in self.populations.keys():
            data = self.get_population_df(population_name)
        else:
            print(f'Invalid population name, must be one of {self.populations.keys()}')
            return None
        xlim, ylim = self.__plot_axis_lims(x=x, y=y, xlim=xlim, ylim=ylim)
        if data.shape[0] < 500:
            ax.scatter(x=data[x], y=data[y], s=3)
            ax = self.__plot_asthetics(ax, x, y, xlim, ylim, title=population_name)
        else:
            self.__standard_2dhist(ax, data, x, y, xlim, ylim, title=population_name)
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
                if p in self.populations.keys():
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
        # TODO NEEDS UPDATING
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

    def population_to_mongo(self, population_name):
        pop_mongo = Population()
        pop_dict = self.populations[population_name]
        for k, v in pop_dict.items():
            if k == 'geom':
                pop_mongo[k] = [(k, v) for k, v in v.items()]
            elif k != 'index':
                pop_mongo[k] = v
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
            FileGroup.objects(primary_id=self.id).update(push__populations=self.population_to_mongo(name))
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
            gating_strategy.gates = list(self.gates.values())
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
            gating_strategy.gates = list(self.gates.values())
            gating_strategy.save()
            exp = FCSExperiment.objects(experiment_id=self.experiment_id).get()
            exp.gating_templates.append(gating_strategy)
            exp.save()
            return True

    def load_template(self, template_name):
        gating_strategy = GatingStrategy.objects(template_name=template_name)
        if gating_strategy:
            self.gates = {gate.gate_name: gate for gate in gating_strategy[0].gates}
            return True
        else:
            print(f'No template with name {template_name}')
            return False
