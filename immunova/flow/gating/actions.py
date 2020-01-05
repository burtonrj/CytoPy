# Dependencies
# Immunova.data
from immunova.data.gating import Gate as DataGate, GatingStrategy
from immunova.data.fcs import FileGroup, Population
from immunova.data.fcs_experiments import FCSExperiment
# Immunova.flow
from immunova.flow.gating.base import GateError
from immunova.flow.gating.static import Static
from immunova.flow.gating.fmo import FMOGate
from immunova.flow.gating.density import DensityThreshold
from immunova.flow.gating.dbscan import DensityBasedClustering
from immunova.flow.gating.quantile import Quantile
from immunova.flow.gating.mixturemodel import MixtureModel
from immunova.flow.gating.transforms import apply_transform
from immunova.flow.gating.defaults import ChildPopulationCollection
from immunova.flow.gating.plotting.static_plots import Plot
from immunova.flow.gating.utilities import get_params, inside_ellipse
# Housekeeping and other tools
from anytree.exporter import DotExporter
from anytree import Node, RenderTree
from anytree.search import findall
from datetime import datetime
from copy import deepcopy
import inspect
import copy
# Scipy
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


class Gating:
    """
    Central class for performing gating on an FCS FileGroup of a single sample
    """
    def __init__(self, experiment: FCSExperiment, sample_id: str, sample: int or None = None,
                 include_controls=True):
        """
        Constructor for Gating
        :param experiment: FCSExperiment currently being processed
        :param sample_id: Identifier of sample of interest
        :param data_type: type of data to load for gating (either 'raw' or 'norm'). Default = 'raw'.
        :param sample: if an integer value is supplied then data will be sampled to this size. Optional (default = None)
        """
        try:
            data = experiment.pull_sample_data(sample_id=sample_id, sample_size=sample,
                                               include_controls=include_controls)
            assert data is not None
        except AssertionError:
            raise GateError(f'Error: failed to fetch data for {sample_id}. Aborting.')
        self.data = [x for x in data if x['typ'] == 'complete'][0]['data']
        if include_controls:
            self.fmo = [x for x in data if x['typ'] == 'control']
            self.fmo = {x['id'].replace(f'{sample_id}_', ''): x['data'] for x in self.fmo}
        else:
            self.fmo = {}
        self.id = sample_id
        self.mongo_id = experiment.fetch_sample_mid(sample_id)
        self.experiment = experiment
        self.plotting = Plot(self)
        self.fmo_search_cache = {_id: dict(root=data.index.values) for _id, data in self.fmo.items()}
        del data
        fg = experiment.pull_sample(sample_id)
        self.filegroup = fg
        self.gates = dict()
        if fg.gates:
            for g in fg.gates:
                self.deserialise_gate(g)

        try:
            self.populations = dict()
            if fg.populations:
                for p in fg.populations:
                    p = p.to_python()
                    parent = p.pop('parent')
                    if parent is not None:
                        parent = self.populations[parent]
                    self.populations[p['name']] = Node(**p, parent=parent)
            else:
                root = Node(name='root', prop_of_parent=1.0, prop_of_total=1.0,
                            warnings=[], geom=dict(shape=None, x='FSC-A', y='SSC-A'), index=self.data.index.values,
                            parent=None)
                self.populations['root'] = root
        except KeyError as e:
            print(f'WARNING: was unable to load populations due to missing parent populations: {e}')
            print('Continuing with blank Gating object. Check that populations have not been removed.')

    def clear_gates(self):
        self.gates = dict()

    def fetch_geom(self, population):
        assert population in self.populations.keys(), f'Population invalid, valid population names: ' \
                                                      f'{self.populations.keys()}'
        return copy.deepcopy(self.populations[population].geom)

    def population_size(self, population: str):
        assert population in self.populations.keys(), f'Population invalid, valid population names: ' \
                                                      f'{self.populations.keys()}'
        return len(self.populations[population].index)

    def deserialise_gate(self, gate):
        kwargs = {k: v for k, v in gate.kwargs}
        kwargs['child_populations'] = ChildPopulationCollection(json_dict=kwargs['child_populations'])
        gate.kwargs = [[k, v] for k, v in kwargs.items()]
        self.gates[gate.gate_name] = gate

    @staticmethod
    def serailise_gate(gate):
        gate = deepcopy(gate)
        kwargs = {k: v for k, v in gate.kwargs}
        kwargs['child_populations'] = kwargs['child_populations'].serialise()
        gate.kwargs = [[k, v] for k, v in kwargs.items()]
        return gate

    @property
    def gating_classes(self) -> dict:
        """
        Available gating classes
        :return: Class look-up dictionary
        """
        available_classes = [Static, FMOGate, DensityBasedClustering, DensityThreshold, Quantile, MixtureModel]
        return {x.__name__: x for x in available_classes}

    def get_population_df(self, population_name: str, transform: bool = False,
                          transform_method: str = 'logicle',
                          transform_features: list or str = 'all') -> pd.DataFrame or None:
        """
        Retrieve a population as a pandas dataframe
        :param population_name: name of population to retrieve
        :param transform: if True, the provided transformation method will be applied to the returned dataframe
        (default = False)
        :param transform_method: transformation method to apply, default = 'logicle' (ignored if transform is False)
        :param transform_features: argument specifying which columns to transform in the returned dataframe. Can either
        be a string value of 'all' (transform all columns), 'fluorochromes' (transform all columns corresponding to a
        fluorochrome) or a list of valid column names
        :return: Population dataframe
        """
        if population_name not in self.populations.keys():
            print(f'Population {population_name} not recognised')
            return None
        idx = self.populations[population_name].index
        data = self.data.loc[idx]
        if transform:
            return apply_transform(data, features_to_transform=transform_features, transform_method=transform_method)
        return data

    def valid_populations(self, populations: list):
        valid = list()
        for pop in populations:
            if pop not in self.populations.keys():
                print(f'Error: {pop} is not a valid population')
            else:
                valid.append(pop)
        return valid

    def labelled_data(self, root_population, labels, transform=False, transform_method: str = 'logicle',
                      transform_features: list or str = 'all'):
        data = self.get_population_df(root_population, transform=transform, transform_method=transform_method,
                                      transform_features=transform_features)
        data['label'] = 'None'
        for x in labels:
            if x in data.columns:
                c = f'{x}_population'
            else:
                c = x
            try:
                data['label'] = data['label'].mask(data.index.isin(self.populations[x].index), c)
            except KeyError:
                print(f'Error: {x} is not a recognised population name')
        return data

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
        # Check cache if this population has been derived previously
        cache_idx = self.search_fmo_cache(target_population, fmo)
        if cache_idx is not None:
            return self.fmo[fmo].loc[cache_idx]
        else:
            cache_idx = self.fmo_search_cache[fmo]['root']

        node = self.populations[target_population]
        route = [x.name for x in node.path][1:]

        # Find start position by searching cache
        for i, pop in enumerate(route[::-1]):
            if pop in self.fmo_search_cache.keys():
                route = route[::-1][:i+1][::-1]
                cache_idx = self.populations[pop].index
                break

        fmo_data = self.fmo[fmo].loc[cache_idx]
        # Predict FMO index
        for pop in route:
            fmo_data = self.fmo[fmo].loc[cache_idx]

            # Train KNN from whole panel data
            x = self.populations[pop].geom['x']
            y = self.populations[pop].geom['y'] or 'FSC-A'

            parent = self.populations[pop].parent.name
            train = self.get_population_df(parent)[[x, y]].copy()
            train['pos'] = 0
            if train.shape[0] > 10000:
                train = train.sample(10000)
            train.pos = train.pos.mask(train.index.isin(self.populations[pop].index), 1)
            y_ = train.pos.values
            knn = KNeighborsClassifier(n_jobs=-1, algorithm='ball_tree', n_neighbors=5)
            knn.fit(train[[x, y]], y_)

            # Predict population in FMO
            y_hat = knn.predict(fmo_data[[x, y]])
            fmo_data['pos'] = y_hat
            cache_idx = fmo_data[fmo_data['pos'] == 1].index.values
            self.fmo_search_cache[fmo][pop] = cache_idx
        return fmo_data.loc[cache_idx]

    @staticmethod
    def __check_class_args(klass, method: str, **kwargs) -> bool:
        """
        Check parameters meet class requirements
        :param klass: Valid gating class
        :param method: Name of class method to be called
        :param kwargs: Keyword arguments supplied by user
        :return: True if valid, else False
        """
        try:
            if not inspect.getmro(klass):
                raise GateError(f'{klass} Invalid: must inherit from Gate class')
            klass_args = get_params(klass, required_only=True, exclude_kwargs=True)
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

    def subtraction(self, target: str, parent: str, new_population_name: str) -> bool:
        """
        Given a target population and a parent population, generate a new population by subtraction of the
        target population from the parent
        :return: True if successful, else False
        """
        if parent not in self.populations.keys():
            print('Error: parent population not recognised')
            return False
        if target not in self.populations.keys():
            print('Error: target population not recognised')
            return False
        x = self.populations[parent].geom['x']
        y = self.populations[parent].geom['y']
        pindex = self.populations[parent].index
        tindex = self.populations[target].index
        index = [p for p in pindex if p not in tindex]
        new_population = ChildPopulationCollection(gate_type='sub')
        new_population.add_population(new_population_name)
        new_population.populations[new_population_name].update_geom(x=x, y=y, shape='sub')
        new_population.populations[new_population_name].update_index(index)
        self.update_populations(output=new_population, parent_df=self.get_population_df(parent),
                                parent_name=parent, warnings=[])
        return True

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

    def __construct_class_and_gate(self, gatedoc: DataGate, kwargs: dict, feedback: bool = True):
        """
        Construct a gating class object and apply the intended method for gating
        :param gatedoc: Gate document generated with `create_gate`
        :param kwargs: keyword arguments for constructor and method
        :return: None
        """
        klass = self.gating_classes[gatedoc.class_]
        parent_population = self.get_population_df(gatedoc.parent)
        expected_const_args = get_params(klass)
        constructor_args = {k: v for k, v in kwargs.items()
                            if k in expected_const_args}
        method_args = {k: v for k, v in kwargs.items()
                       if k in inspect.signature(getattr(klass, gatedoc.method)).parameters.keys()}
        analyst = klass(data=parent_population, **constructor_args)
        output = getattr(analyst, gatedoc.method)(**method_args)
        if feedback:
            print(f'------ {gatedoc.gate_name} ------')
            if analyst.warnings:
                for x in analyst.warnings:
                    print(x)
        self.update_populations(output, parent_df=parent_population,
                                warnings=analyst.warnings, parent_name=gatedoc.parent)
        if feedback:
            for pop in output.populations.keys():
                print(f'New population: {pop}')
                print(f'...proportion of total events: {self.populations[pop].prop_of_total:.3f}')
                print(f'...proportion of parent: {self.populations[pop].prop_of_parent:.3f}')
            print('-----------------------')

    def apply(self, gate_name: str, plot_output: bool = True, feedback: bool = True, **kwargs) -> None:
        """
        Apply a gate to events data (must be generated with `create_gate` first)
        :param gate_name: Name of the gate to apply
        :param plot_output: If True, resulting gates will be printed to screen
        :return: None
        """
        gatedoc = self.__apply_checks(gate_name)
        if gatedoc is None:
            return None
        gkwargs = {k: v for k, v in gatedoc.kwargs}
        # Alter kwargs if given
        for k, v in kwargs.items():
            gkwargs[k] = v
        if 'fmo_x' in gkwargs.keys():
            gkwargs['fmo_x'] = self.get_fmo_data(gatedoc.parent, gkwargs['fmo_x'])
        if 'fmo_y' in kwargs.keys():
            gkwargs['fmo_y'] = self.get_fmo_data(gatedoc.parent, gkwargs['fmo_y'])
        self.__construct_class_and_gate(gatedoc, gkwargs, feedback)
        if plot_output:
            self.plotting.plot_gate(gate_name=gate_name)

    def update_populations(self, output: ChildPopulationCollection, parent_df: pd.DataFrame, warnings: list,
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

    def __update_index(self, population_name: str, geom: dict):
        def transform_x_y(d):
            d = d.copy()
            assert all([t in geom.keys() for t in ['transform_x', 'transform_y']]), 'Geom must contain a key "transform_x", the transform method for the x-axis AND ' \
                                                                                    'a key "transform_y", the transform method for the y-axis'
            if geom['transform_x'] is not None:
                d = apply_transform(d, transform_method=geom['transform_x'], features_to_transform=[geom['x']])
            if geom['transform_y'] is not None:
                d = apply_transform(d, transform_method=geom['transform_y'], features_to_transform=[geom['y']])
            return d

        assert population_name in self.populations.keys(), f'Population {population_name} does not exist'
        assert 'definition' in geom.keys(), 'Geom must contain key "definition", a string value that indicates ' \
                                            'if population is the "positive" or "negative"'
        parent_name = self.populations[population_name].parent.name
        parent = self.get_population_df(parent_name, transform=False)
        if geom['shape'] == 'threshold':
            assert 'threshold' in geom.keys(), 'Geom must contain a key "threshold" with a float value'
            assert 'transform_x' in geom.keys(), 'Geom must contain a key "transform_x", the transform method for the x-axis'
            if geom['transform_x'] is not None:
                parent = apply_transform(parent, transform_method=geom['transform_x'], features_to_transform=[geom['x']])
            if geom['definition'] == '+':
                return parent[parent[geom['x']] >= geom['threshold']].index.values
            if geom['definition'] == '-':
                return parent[parent[geom['x']] < geom['threshold']].index.values
            raise ValueError('Definition must have a value of "+" or "-" for a 1D threshold gate')
        if geom['shape'] == '2d_threshold':
            def geom_bool(definition, p, x_, y_):
                if definition == '++':
                    return p[x_ & y_].index.values
                if definition == '--':
                    return p[~(x_ & y_)].index.values
                if definition == '+-':
                    return p[x_ & (~y_)].index.values
                if definition == '-+':
                    return p[(~x_) & y_].index.values
                raise ValueError('Definition must have a value of "+-", "-+", "--", or "++" for a 2D threshold gate')

            assert all([t in geom.keys() for t in ['threshold_x', 'threshold_y']]), 'Geom must contain keys "threshold_x" and "threshold_y" both with a float value'
            parent = transform_x_y(parent)
            x = parent[geom['x']].round(decimals=2) >= round(geom['threshold_x'], 2)
            y = parent[geom['y']].round(decimals=2) >= round(geom['threshold_y'], 2)
            if type(geom['definition']) == list:
                idx = list(map(lambda d: geom_bool(d, parent, x, y), geom['definition']))
                return [i for l in idx for i in l]
            else:
                return geom_bool(geom['definition'], parent, x, y)

        if geom['shape'] == 'rect':
            keys = ['x_min', 'x_max', 'y_min', 'y_max']
            assert all([r in geom.keys() for r in keys]), f'Geom must contain keys {keys} both with a float value'
            parent = transform_x_y(parent)
            x = (parent[geom['x']] >= geom['x_min']) & (parent[geom['x']] <= geom['x_max'])
            y = (parent[geom['y']] >= geom['y_min']) & (parent[geom['y']] <= geom['y_max'])
            pos = parent[x & y]
            if geom['definition'] == '+':
                return pos.index.values
            if geom['definition'] == '-':
                return parent[~parent.index.isin(pos.index)].index.values
            raise ValueError('Definition must have a value of "+" or "-" for a rectangular geom')
        if geom['shape'] == 'ellipse':
            keys = ['centroid', 'width', 'height', 'angle']
            assert all([c in geom.keys() for c in keys]), f'Geom must contain keys {keys}; note, centroid must be a tuple and all others a float value'
            parent = transform_x_y(parent)
            channels = [geom['x'], geom['y']]
            mask = inside_ellipse(parent[channels].values,
                                  center=tuple(geom['centroid']),
                                  width=geom['width'],
                                  height=geom['height'],
                                  angle=geom['angle'])
            pos = parent[mask]
            if geom['definition'] == '+':
                return pos.index
            if geom['definition'] == '-':
                return parent[~parent.index.isin(pos.index)].index.values
            raise ValueError('Definition must have a value of "+" or "-" for a ellipse geom')
        raise ValueError('Gates producing polygon geoms (i.e. like density based clustering), substitution gates '
                         'and supervised ML gates cannot be edited in the current build of Immunova.')

    def edit_gate(self, gate_name: str, updated_geom: dict, delete=True):
        """
        Manually replace the outcome of a gate by updating the geom of it's child populations.
        :param gate_name:
        :param updated_geom:
        :param delete:
        :return:
        """
        print(f'Editing gate: {gate_name}')
        assert gate_name in self.gates.keys(), f'Invalid gate, existing gates are: {self.gates.keys()}'
        children = self.gates[gate_name].children
        effected_populations = list()
        immediate_children = list()
        for c in children:
            print(f'Updating {c}')
            assert c in updated_geom.keys(), f'Invalid child populations specified/missing child, gate {gate_name} ' \
                                             f'has the following children: {children}'
            self.populations[c].geom = updated_geom[c]
            self.populations[c].index = self.__update_index(c, updated_geom[c])
            effected_populations = effected_populations + self.find_dependencies(population=c)
            immediate_children = immediate_children + [n.name for n in self.populations[c].children]
        effected_gates = [name for name, gate in self.gates.items() if gate.parent in effected_populations]
        print(f'The following gates are downstream of {gate_name} and will need to be applied again: {effected_gates}')
        if delete:
            for c in immediate_children:
                self.remove_population(c)
        print('Edit complete!')

    def find_dependencies(self, population: str = None) -> list or None:
        """
        For a given population find all dependencies
        :param population: population name
        :return: List of populations dependent on given population
        """
        if population not in self.populations.keys():
            print(f'Error: population {population} does not exist; '
                  f'valid population names include: {self.populations.keys()}')
            return None
        root = self.populations['root']
        node = self.populations[population]
        return [x.name for x in findall(root, filter_=lambda n: node in n.path)]

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
            dependencies = self.find_dependencies(population=child)
            if dependencies is None:
                continue
            effected_populations = effected_populations + dependencies
            self.remove_population(child)
            effected_populations.append(child)
        effected_gates = [name for name, gate in self.gates.items() if gate.parent in effected_populations]
        effected_gates.append(gate_name)
        for g in effected_gates:
            self.gates.pop(g)
        return effected_gates, effected_populations

    def print_population_tree(self, image: bool = False, image_name: str or None = None) -> None:
        """
        Generate a tree diagram of the populations associated to this Gating object and print to stdout
        :param image: if True, an image will be saved to the working directory
        :param image_name: name of the resulting image file, ignored if image = False (optional; default name is of
        format `filename_populations.png`
        :return: None
        """
        root = self.populations['root']
        if image:
            if image_name is None:
                image_name = f'{self.id}_population_tree.png'
            DotExporter(root).to_picture(image_name)
        for pre, fill, node in RenderTree(root):
            print('%s%s' % (pre, node.name))

    def population_to_mongo(self, population_name: str) -> Population:
        """
        Convert a population into a mongoengine Population document
        :param population_name: Name of population to convert
        :return: Population document
        """
        pop_node = self.populations[population_name]
        if pop_node.geom is None:
            geom = []
        else:
            geom = [(k, v) for k, v in pop_node.geom.items()]

        parent = None
        if pop_node.parent:
            parent = pop_node.parent.name
        pop_mongo = Population(population_name=pop_node.name,
                               parent=parent,
                               prop_of_parent=pop_node.prop_of_parent,
                               prop_of_total=pop_node.prop_of_total,
                               warnings=pop_node.warnings,
                               geom=geom)
        pop_mongo.save_index(pop_node.index)
        return pop_mongo

    def clean(self, population: str, qt: float = 0.999, qb: float = 0.001):
        if population not in self.populations.keys():
            raise KeyError(f'No such population {population}')
        data = self.get_population_df(population, transform=True)
        qt = data.apply(lambda x: x <= x.quantile(qt), axis=0)
        qb = data.apply(lambda x: x >= x.quantile(qb), axis=0)
        clean = data[qt].dropna()[qb].dropna()
        geom = {'method': 'Clean',
                'x': '',
                'y': ''}
        prop_of_parent = clean.shape[0]/data.shape[0]
        prop_of_total = len(self.populations['root'].index) / data.shape[0]
        self.populations[f'{population}_cleaned'] = Node(name=f'{population}_cleaned',
                                                         population_name=f'{population}_cleaned',
                                                         index=clean.index,
                                                         prop_of_parent=prop_of_parent,
                                                         prop_of_total=prop_of_total,
                                                         geom=geom, warnings=[],
                                                         parent=self.populations[population])

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
            FileGroup.objects(id=self.mongo_id).update(push__populations=self.population_to_mongo(name))
        for _, gate in self.gates.items():
            gate = self.serailise_gate(gate)
            FileGroup.objects(id=self.mongo_id).update(push__gates=gate)
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
            gating_strategy.gates = [self.serailise_gate(gate) for gate in list(self.gates.values())]
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
            gating_strategy.gates = [self.serailise_gate(gate) for gate in list(self.gates.values())]
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
            for gate in gating_strategy[0].gates:
                self.deserialise_gate(gate)
            return True
        else:
            print(f'No template with name {template_name}')
            return False
