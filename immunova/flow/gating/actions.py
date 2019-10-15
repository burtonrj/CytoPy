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
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.plotting.static_plots import Plot
# Housekeeping and other tools
from anytree import Node
from anytree.search import findall
from datetime import datetime
import inspect
# Scipy
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np


class Gating:
    """
    Central class for performing gating on an FCS FileGroup of a single sample
    """
    def __init__(self, experiment: FCSExperiment, sample_id: str,
                 data_type='raw', sample: int or None = None):
        """
        Constructor for Gating
        :param experiment: FCSExperiment currently being processed
        :param sample_id: Identifier of sample of interest
        :param data_type: type of data to load for gating (either 'raw' or 'norm'). Default = 'raw'.
        :param sample: if an integer value is supplied then data will be sampled to this size. Optional (default = None)
        """
        try:
            data = experiment.pull_sample_data(sample_id=sample_id, data_type=data_type, sample_size=sample)
            assert data is not None
            self.data = [x for x in data if x['typ'] == 'complete'][0]['data']
            self.fmo = [x for x in data if x['typ'] == 'control']
            self.fmo = {x['id']: x['data'] for x in self.fmo}
            self.id = sample_id
            self.experiment = experiment
            self.plotting = Plot(self)
            self.fmo_search_cache = {_id: dict(root=data.index.values) for _id, data in self.fmo.items()}
            del data

            fg = experiment.pull_sample(sample_id)
            self.filegroup = fg
            self.gates = dict()
            if fg.gates:
                for g in fg.gates:
                    self.gates[g.gate_name] = g

            self.populations = dict()
            if fg.populations:
                for p in fg.populations:
                    self.populations[p.population_name] = p.to_node()
            else:
                root = Node(name='root', prop_of_parent=1.0, prop_of_total=1.0,
                            warnings=[], geom=dict(shape=None, x='FSC-A', y='SSC-A'), index=self.data.index.values,
                            parent=None)
                self.populations['root'] = root
        except AssertionError:
            print('Error: failed to construct Gating object')

    @property
    def gating_classes(self) -> dict:
        """
        Available gating classes
        :return: Class look-up dictionary
        """
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
        idx = self.populations[population_name].index
        return self.data.loc[idx]

    def search_fmo_cache(self, target_population, fmo):
        if target_population in self.fmo_search_cache[fmo].keys():
            return self.fmo_search_cache[fmo][target_population]
        return None

    def get_fmo_data(self, target_population, fmo):
        """
        Calculate population of fmo data using supervised machine learning and primary data as training set
        :param target_population:
        :param fmo:
        :return:
        """
        cache_idx = self.search_fmo_cache(target_population, fmo)
        if cache_idx is not None:
            return self.fmo[fmo].loc[cache_idx]
        root = self.populations['root']
        node = self.populations[target_population]
        route = findall(root, filter_=lambda n: node in root.path)
        # Search route for start position
        start = 0
        for pop in route[::-1]:
            cache_idx = self.search_fmo_cache(pop.name, fmo)
            if cache_idx is not None:
                for idx, pop in enumerate(route):
                    if pop.name == pop.name:
                        start = idx
                        break
                break
        # Predict FMO index
        route = route[start:]
        fmo_data = self.fmo[fmo].loc[cache_idx]
        for pop in route:
            fmo_data = self.fmo[fmo].loc[cache_idx]
            x = pop.geom['x']
            y = pop.geom['y'] or 'FSC-A'
            train = self.get_population_df(pop.name)[[x, y]].copy()
            train['pos'] = 0
            if train.shape[0] > 10000:
                train = train.sample(10000)
            train.pos = train.pos.mask(train.index.isin(pop.index), 1)
            y_ = train.pos.values
            knn = KNeighborsClassifier(n_jobs=-1, algorithm='kd_tree', n_neighbors=5, metric='manhattan')
            knn.fit(train, y_)
            y_hat = knn.predict(fmo_data[[x, y]])
            fmo_data['pos'] = y_hat
            cache_idx = fmo_data[fmo_data['pos'] == 1].index.values
            self.fmo_search_cache[fmo][pop.name] = cache_idx
        return fmo_data.loc[cache_idx]

    @staticmethod
    def __check_class_args(klass: Gate, method: str, **kwargs) -> bool:
        """
        Check parameters meet class requirements
        :param klass: Valid gating class
        :param method: Name of class method to be called
        :param kwargs: Keyword arguments supplied by user
        :return: True if valid, else False
        """
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
                if arg == 'self':
                    continue
                if arg not in kwargs.keys():
                    print(f'Error: missing required method argument {arg} for method '
                          f'{method} belonging to {klass.__name__}')
                    return False
            return True
        except AttributeError:
            print(f'Error: {method} is not a valid method for class {klass.__name__}')
            return False

    def create_gate(self, gate_name: str, parent: str, class_: str, method: str, kwargs: dict,
                    child_populations: ChildPopulationCollection) -> bool:
        """
        Create a gate
        :param gate_name: Name of the gate
        :param parent: Name of parent population gate is applied to
        :param class_: Name of a valid gating class
        :param method: Name of the class method to invoke upon gating
        :param kwargs: Keyword arguments (include constructor arguments and method arguments)
        :param child_populations: A valid ChildPopulationCollection object describing the resulting populations
        :return: True if successful, else False
        """
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

    def __apply_checks(self, gate_name: str) -> DataGate or None:
        """
        Default checks applied whenever a gate is applied
        :param gate_name: Name of gate to apply
        :return: Gate document (None if checks fail)
        """
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

    def __construct_class_and_gate(self, gatedoc: DataGate, kwargs: dict):
        """
        Construct a gating class object and apply the intended method for gating
        :param gatedoc: Gate document generated with `create_gate`
        :param kwargs: keyword arguments for constructor and method
        :return: None
        """
        klass = self.gating_classes[gatedoc.class_]
        parent_population = self.get_population_df(gatedoc.parent)
        constructor_args = {k: v for k, v in kwargs.items() if k in inspect.signature(klass).parameters.keys()}
        method_args = {k: v for k, v in kwargs.items() if k in inspect.signature(getattr(klass, gatedoc.method)).parameters.keys()}
        analyst = klass(data=parent_population, **constructor_args)
        output = getattr(analyst, gatedoc.method)(**method_args)
        self.__update_populations(output, parent_df=parent_population,
                                  warnings=analyst.warnings, parent_name=gatedoc.parent)

    def apply(self, gate_name: str, plot_output: bool = True) -> None:
        """
        Apply a gate to events data (must be generated with `create_gate` first)
        :param gate_name: Name of the gate to apply
        :param plot_output: If True, resulting gates will be printed to screen
        :return: None
        """
        gatedoc = self.__apply_checks(gate_name)
        if gatedoc is None:
            return None
        kwargs = {k: v for k, v in gatedoc.kwargs}
        if 'fmo_x' in kwargs.keys():
            kwargs['fmo_x'] = self.get_fmo_data(gatedoc.parent, kwargs['fmo_x'])
        if 'fmo_y' in kwargs.keys():
            kwargs['fmo_y'] = self.get_fmo_data(gatedoc.parent, kwargs['fmo_y'])
        self.__construct_class_and_gate(gatedoc, kwargs)
        if plot_output:
            self.plotting.plot_gate(gate_name=gate_name)

    def __update_populations(self, output: ChildPopulationCollection, parent_df: pd.DataFrame, warnings: list,
                             parent_name: str):
        """
        Given some ChildPopulationCollection object generated from a gate, update saved populations
        :param output: ChildPopulationCollection object generated from a gate
        :param parent_df: pandas dataframe of events data from parent population
        :param warnings: list of warnings generated from gate
        :param parent_name: name of the parent population
        :return:
        """
        for name, population in output.populations.items():
            n = len(population.index)
            if n == 0:
                prop_of_total = 0
                prop_of_parent = 0
            else:
                prop_of_parent = n / parent_df.shape[0]
                prop_of_total = n / self.data.shape[0]
            geom = None
            if population.geom is not None:
                geom = population.geom.as_dict()
            self.populations[name] = Node(name=name, population_name=name, index=population.index,
                                          prop_of_parent=prop_of_parent,
                                          prop_of_total=prop_of_total,
                                          geom=geom, warnings=warnings,
                                          parent=self.populations[parent_name])
        return output

    def apply_many(self, gates: list = None, apply_all=False, plot_outcome=False, feedback=True):
        """
        Apply multiple existing gates sequentially
        :param gates: Name of gates to apply (NOTE: Gates must be provided in sequential order!)
        :param apply_all: If True, gates is ignored and all current gates will be applied
        (population tree must be empty)
        :param plot_outcome: If True, resulting gates will be printed to screen
        :param feedback: If True, feedback will be printed to stdout
        :return: None
        """
        if gates is None:
            gates = list()
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
            self.apply(gate_name, plot_output=plot_outcome)
        if feedback:
            print('Complete!')

    def find_dependencies(self, population: str = None) -> list or None:
        """
        For a given population find all dependencies
        :param population: population name
        :return: List of populations dependent on given population
        """
        if population not in self.populations.keys():
            print(f'Population {population} does not exist')
            return None
        root = self.populations['root']
        node = self.populations[population]
        dependent_paths = findall(root, filter_=lambda n: node in root.path)
        dependencies = [name for name, node in self.populations.items() if node in dependent_paths]
        return dependencies

    def remove_population(self, population_name: str) -> None:
        """
        Remove a population
        :param population_name: name of population to remove
        :return: None
        """
        if population_name not in self.populations.keys():
            print(f'{population_name} does not exist')
            return None
        downstream_populations = self.find_dependencies(population=population_name)
        self.populations[population_name].parent = None
        for x in downstream_populations:
            self.populations.pop(x)

    def remove_gate(self, gate_name: str, propagate: bool = True) -> list and list or None:
        """
        Remove gate
        :param gate_name: name of gate to remove
        :param propagate: If True, downstream gates and effected populations will also be removed
        :return: list of removed gates, list of removed populations
        """
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
        return effected_gates, effected_populations

    def population_to_mongo(self, population_name: str) -> Population:
        """
        Convert a population into a mongoengine Population document
        :param population_name: Name of population to convert
        :return: Population document
        """
        pop_node = self.populations[population_name]
        geom = [(k, v) for k, v in pop_node.geom.items()]
        pop_mongo = Population(population_name=pop_node.name,
                               parent=pop_node.parent.name,
                               prop_of_parent=pop_node.prop_of_parent,
                               prop_of_total=pop_node.prop_of_total,
                               warnings=pop_node.warnings,
                               geom=geom)
        pop_mongo.save_index(pop_node.index)
        return pop_mongo

    def save(self, overwrite=False) -> bool:
        """
        Save all gates and population's to mongoDB
        :param overwrite: If True, existing populations/gates for sample will be overwritten
        :return: True if successful else False
        """
        fg = self.filegroup
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
    """
    Generate a reusable template for gating. Inherits all functionality of Gating class.
    """
    def save_new_template(self, template_name: str, overwrite: bool = True) -> bool:
        """
        Save template structure
        :param template_name: name of the template
        :param overwrite: If True, any existing template with the same name will be overwritten
        :return: True if successful, else False
        """
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
            templates = [x for x in self.experiment.gating_templates
                         if x.template_name != gating_strategy.template_name]
            templates.append(gating_strategy)
            self.experiment.gating_templates = templates
            self.experiment.save()
            return True
        else:
            print(f'No existing template named {template_name}, creating new template')
            gating_strategy = GatingStrategy()
            gating_strategy.template_name = template_name
            gating_strategy.creation_date = datetime.now()
            gating_strategy.last_edit = datetime.now()
            gating_strategy.gates = list(self.gates.values())
            gating_strategy.save()
            self.experiment.gating_templates.append(gating_strategy)
            self.experiment.save()
            return True

    def load_template(self, template_name: str) -> bool:
        """
        Load gates from a template
        :param template_name: name of template to load
        :return: True if successful, else False
        """
        gating_strategy = GatingStrategy.objects(template_name=template_name)
        if gating_strategy:
            self.gates = {gate.gate_name: gate for gate in gating_strategy[0].gates}
            return True
        else:
            print(f'No template with name {template_name}')
            return False
