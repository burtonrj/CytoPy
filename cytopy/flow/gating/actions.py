# Dependencies
# Immunova.data
from ...data.gating import Gate as DataGate, GatingStrategy
from ...data.fcs import FileGroup, Population, ControlIndex
from ...data.fcs_experiments import FCSExperiment
# Immunova.flow
from ..transforms import apply_transform
from .base import GateError
from .static import Static
from .control import ControlGate
from .density import DensityThreshold
from .dbscan import DensityBasedClustering
from .quantile import Quantile
from .mixturemodel import MixtureModel
from .defaults import ChildPopulationCollection
from .plotting.static_plots import Plot
from .utilities import get_params, inside_ellipse, inside_polygon
from ..feedback import progress_bar
# Housekeeping and other tools
from anytree.exporter import DotExporter
from anytree import Node, RenderTree
from anytree.search import findall
from scipy.spatial import ConvexHull
from shapely.geometry.polygon import Polygon
from datetime import datetime
from copy import deepcopy
import inspect
import copy
# Scipy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np


class Gating:
    """Central class for performing semi-automated gating and storing gating information on an FCS FileGroup of a single sample.
    
    Parameters
    -----------
    experiment: FCSExperiment
        experiment you're currently working on
    sample_id: str
        name of the sample to analyse (must belong to experiment)
    sample: int, optional
        number of events to sample from FCS file(s) (optional)
    include_controls: bool, (default=True)
        if True and FMOs are included for specified samples, the FMO data will also be loaded into the Gating object
    default_axis: str, (default='FSC-A')
        default value for y-axis for all plots
    """
    def __init__(self,
                 experiment: FCSExperiment,
                 sample_id: str,
                 sample: int or None = None,
                 include_controls=True,
                 default_axis='FSC-A'):
        try:
            data = experiment.pull_sample_data(sample_id=sample_id,
                                               sample_size=sample,
                                               include_controls=include_controls)
            assert data is not None
        except AssertionError:
            raise GateError(f'Error: failed to fetch data for {sample_id}. Aborting.')
        self.data = [x for x in data
                     if x['typ'] == 'complete'][0]['data']
        if include_controls:
            self.ctrl = [x for x in data
                         if x['typ'] == 'control']
            self.ctrl = {x['id'].replace(f'{sample_id}_', ''): x['data']
                         for x in self.ctrl}
        else:
            self.ctrl = {}
        self.id = sample_id
        self.mongo_id = experiment.fetch_sample_mid(sample_id)
        self.experiment = experiment
        self.plotting = Plot(self, default_axis)
        del data
        fg = experiment.pull_sample(sample_id)
        self.filegroup = fg
        self.gates = dict()
        self.populations = dict()
        if fg.gates:
            for g in fg.gates:
                self._deserialise_gate(g)

        self.populations = self._construct_tree(fg=fg)

    @staticmethod
    def _construct_unordered(tree: dict, new_population: dict):
        parent = new_population.get('parent')
        if parent is None:
            tree[new_population.get('name')] = Node(parent=None, **new_population)
            return tree
        if parent not in tree.keys():
            return False
        parent = new_population.pop('parent')
        tree[new_population.get('name')] = Node(**new_population, parent=tree.get(parent))
        return tree

    def _construct_tree(self,
                        fg: FileGroup):
        """
        Internal function. Called on instantiation and constructs population tree.

        Parameters
        ----------
        fg: FileGroup
            Sample FileGroup object
        Returns
        -------
        Dict
            Returns dictionary of population Nodes
        """
        populations = dict()
        if not fg.populations or len(fg.populations) == 1:
            control_cache = None
            if self.ctrl:
                control_cache = {control_id: control_data.index.values for
                                 control_id, control_data in self.ctrl.items()}
            populations['root'] = Node(name='root', prop_of_parent=1.0, prop_of_total=1.0,
                                       warnings=[], geom=dict(shape=None, x='FSC-A', y='SSC-A'),
                                       index=self.data.index.values,
                                       parent=None, control_idx=control_cache)
            return populations

        # Reconstruct tree and populate control cache is necessary
        root = fg.get_population('root').to_python()
        root.pop('parent')
        populations['root'] = Node(**root, parent=None)
        database_populations = [p.to_python() for p in fg.populations if p.population_name != 'root']
        i = 0
        while len(database_populations) > 0:
            if i >= len(database_populations):
                i = 0
            tree = self._construct_unordered(populations, database_populations[i])
            if tree:
                populations = tree
                database_populations = [p for p in database_populations
                                        if p.get('name') != database_populations[i].get('name')]
            else:
                i = i + 1

        if self.ctrl:
            if not populations['root'].control_idx:
                populations['root'].control_idx = {control_id: control_data.index.values for
                                                   control_id, control_data in self.ctrl.items()}
        return populations

    def clear_gates(self):
        """Remove all currently associated gates."""
        self.gates = dict()

    def fetch_geom(self,
                   population: str) -> dict:
        """Given the name of a population, retrieve the geom that defined this population

        Parameters
        ----------
        population : str
            name of population to be fetched

        Returns
        -------
        dict
            Population geom (dictionary)
        """
        assert population in self.populations.keys(), f'Population invalid, valid population names: ' \
                                                      f'{self.populations.keys()}'
        return copy.deepcopy(self.populations[population].geom)

    def population_size(self,
                        population: str):
        """
        Returns in integer count for the number of events in a given population

        Parameters
        ----------
        population : str
            population name

        Returns
        -------
        int
            event count
        """
        assert population in self.populations.keys(), f'Population invalid, valid population names: ' \
                                                      f'{self.populations.keys()}'
        return len(self.populations[population].index)

    def _deserialise_gate(self,
                          gate: DataGate):
        """
        Given some Gate document from the database, deserialise for use; re-instantiate ChildPopulationCollection

        Parameters
        ----------
        gate : Gate
            Gate object to deserialise

        Returns
        -------
        None
        """
        kwargs = {k: v for k, v in gate.kwargs}
        kwargs['child_populations'] = ChildPopulationCollection(json_dict=kwargs['child_populations'])
        gate.kwargs = [[k, v] for k, v in kwargs.items()]
        self.gates[gate.gate_name] = gate

    @staticmethod
    def _serailise_gate(gate: DataGate):
        """
        Given some Gate document, serialise so that it can be saved to the database

        Parameters
        ----------
        gate : Gate
            Gate object to serialise

        Returns
        -------
        Gate
            New 'database friendly' Gate
        """
        gate = deepcopy(gate)
        kwargs = {k: v for k, v in gate.kwargs}
        kwargs['child_populations'] = kwargs['child_populations'].serialise()
        gate.kwargs = [[k, v] for k, v in kwargs.items()]
        return gate

    @property
    def gating_classes(self) -> dict:
        """
        Available gating classes

        Returns
        -------
        dict
            Class look-up dictionary
        """
        available_classes = [Static, ControlGate, DensityBasedClustering, DensityThreshold, Quantile, MixtureModel]
        return {x.__name__: x for x in available_classes}

    def get_population_df(self,
                          population_name: str,
                          transform: bool = False,
                          transform_method: str or None = 'logicle',
                          transform_features: list or str = 'all',
                          label: bool = False,
                          ctrl_id: str or None = None) -> pd.DataFrame or None:
        """
        Retrieve a population as a pandas dataframe

        Parameters
        ----------
        population_name : str
            name of population to retrieve
        transform : bool, (default=False)
            if True, the provided transformation method will be applied to the returned dataframe
        transform_method : str, (default='logicle')
            transformation method to apply, default = 'logicle' (ignored if transform is False)
        transform_features : list or str, (default='all')
            argument specifying which columns to transform in the returned dataframe. Can either
            be a string value of 'all' (transform all columns), 'fluorochromes' (transform all columns corresponding to a
            fluorochrome) or a list of valid column names
        label: bool, (default=False)
            If true, additional column included named 'label' with population label (terminal node in population tree)
        ctrl_id: str, optional
            If given, retrieves DataFrame of data from control file rather than primary data

        Returns
        -------
        Pandas.DataFrame or None
            Population DataFrame

        """
        if population_name not in self.populations.keys():
            print(f'Population {population_name} not recognised')
            return None
        if ctrl_id is None:
            idx = self.populations[population_name].index
            data = self.data.loc[idx]
        else:
            idx = self.populations[population_name].control_idx.get(ctrl_id)
            assert idx is not None, f'No cached index for {ctrl_id} associated to population {population_name}, ' \
                                    f'have you called "control_gating" previously?'
            data = self.ctrl[ctrl_id].loc[idx]
        if label:
            data['label'] = None
            dependencies = self.find_dependencies(population_name)
            dependencies = [p for p in dependencies if p != population_name]
            for pop in dependencies:
                idx = self.populations[pop].index
                data.loc[idx, 'label'] = pop
        if transform_method is None:
            transform = False
        if transform:
            return apply_transform(data, features_to_transform=transform_features, transform_method=transform_method)
        return data

    def valid_populations(self,
                          populations: list,
                          verbose: bool = True):
        """
        Given a list of populations, check validity and return list of valid populations

        Parameters
        ----------
        populations : list
            list of populations to check
        verbose : bool, (default=True)
            if True, prints invalid population

        Returns
        -------
        List
            Valid populations
        """
        valid = list()
        for pop in populations:
            if pop not in self.populations.keys():
                if verbose:
                    print(f'{pop} is not a valid population')
            else:
                valid.append(pop)
        return valid

    def search_ctrl_cache(self,
                          target_population: str,
                          ctrl_id: str,
                          return_dataframe: bool = False):
        """Search the control cache for a target population. Return either index of events belonging to target
        population in control data or Pandas DataFrame of target population from control data.

        Parameters
        ----------
        target_population : str
            Name of population to extract from control data
        ctrl_id : str
            Name of control to extract data from
        return_dataframe : bool, (default=False)
            If True, return a Pandas DataFrame of target population, else return index of events belonging to target
            population

        Returns
        -------
        Pandas.DataFrame or Numpy.array
        """
        assert target_population in self.populations.keys(), f'Invalid population {target_population}'
        assert self.ctrl, 'No control data present for current gating instance, was "include_controls" set to False?'
        assert ctrl_id in self.ctrl.keys(), f'No control data found for {ctrl_id}'
        if return_dataframe:
            idx = self.populations[target_population].control_idx.get(ctrl_id)
            if idx is None:
                raise ValueError(f'No population data found for {ctrl_id}, have you called "control_gating"?')
            return self.ctrl[ctrl_id].loc[idx]
        return self.populations[target_population].control_idx.get(ctrl_id)

    def _predict_ctrl_population(self,
                                 target_population: str,
                                 ctrl_id: str,
                                 model: SVC or KNeighborsClassifier,
                                 mappings: dict or None = None):
        """Internal method. Predict a target population for a given control using the primary data as
        training data. Results are assigned to population node.

        Parameters
        ----------
        target_population : str
            Name of population to predict
        ctrl_id : str
            Name of the control to predict population for
        model : Object
            Instance of Scikit-Learn classifier
        mappings : dict, optional
            Dictionary of axis mappings for classification (necessary if training data is generated from a supervised
            learning method)

        Returns
        -------
        None
        """
        assert target_population in self.populations.keys(), f'Invalid population {target_population}'
        target_node = self.populations.get(target_population)
        cache_idx = self.search_ctrl_cache(target_node.parent.name, ctrl_id)
        if cache_idx is None:
            self._predict_ctrl_population(target_node.parent.name, ctrl_id, model, mappings)
            cache_idx = self.search_ctrl_cache(target_node.parent.name, ctrl_id)

        x, y = target_node.geom.get('x'), target_node.geom.get('y') or 'FSC-A'
        if x is None or y is None:
            assert mappings, f'{target_population} has no specified x and y, please provide mappings'
            x = x or mappings.get('x')
            y = y or mappings.get('y')
        # Prepare training data
        train = self.get_population_df(target_node.parent.name)[[x, y]].copy()
        if train.shape[0] > 10000:
            train = train.sample(10000)
        train['pos'] = 0
        train.pos = train.pos.mask(train.index.isin(target_node.index), 1)
        y_ = train.pos.values
        # Fit model and predict values for fmo
        model.fit(train[[x, y]], y_)
        ctrl_data = self.ctrl.get(ctrl_id).loc[cache_idx]
        ctrl_data['pos'] = model.predict(ctrl_data[[x, y]])
        self.populations[target_population].control_idx[ctrl_id] = ctrl_data[ctrl_data['pos'] == 1].index.values

    def clear_control_cache(self, ctrl_id: str) -> None:
        """
        Remove all associated populations to given control.

        Parameters
        ----------
        ctrl_id : str
            Control to remove populations from

        Returns
        -------
        None
        """
        assert ctrl_id in self.ctrl.keys(), f'No control data found for {ctrl_id}'
        for pop_name in self.populations.keys():
            self.populations[pop_name]['control_idx'].pop(pop_name)

    def control_gating(self, ctrl_id: str,
                       tree_map: dict or None = None,
                       overwrite_existing: bool = False,
                       verbose: bool = True,
                       model: str = 'knn',
                       **model_kwargs) -> None:
        """
        Using the primary data as a training set, transverse over the population tree and predict the same
        populations for some given control. Results of classification are saved to the population nodes.

        Parameters
        ----------
        ctrl_id : str
            Name of control data to predict populations for; must be a valid ID currently associated to Gating object
        tree_map : dict, optional
            Dictionary describing the axis of populations in the population tree. This is only necessary if populations
            currently in population tree were generated using a supervised machine learning method.
        overwrite_existing : bool, (default=False)
            If True, any existing control populations will be removed
        verbose : bool, (default=True)
            If True, a progress bar is provided as well as text output
        model : str, (default='knn')
            Type of model to use for per-population classification. Currently supports K-Nearest Neighbours (knn) and
            Support Vector Machines (svm)
        model_kwargs :
            Additional keyword arguments to pass to instance of Scikit-Learn classifier (see sklearn documentation)

        Returns
        -------
        None
        """
        if tree_map is None:
            tree_map = {}
        vprint = print if verbose else lambda *a, **k: None
        # Check that control data is available
        assert self.ctrl, 'No control data present for current gating instance, was "include_controls" set to False?'
        assert ctrl_id in self.ctrl.keys(), f'No control data found for {ctrl_id}'
        assert all([all([i in x.keys() for i in ['x', 'y']]) for x in tree_map.values()]), 'Invalid tree_map'
        if model == 'knn':
            model = KNeighborsClassifier(**model_kwargs)
        elif model == 'svm':
            model = SVC(**model_kwargs)
        else:
            raise ValueError('Currently only KNearestNeighbours (knn) or Support Vector Machines (svm) are supported')

        if overwrite_existing:
            self.clear_control_cache(ctrl_id)
        sml_populations = any([p.geom.get('shape') == 'sml' for p in self.populations.values()])
        if sml_populations:
            assert tree_map, 'One or more gates detected that have been generated by a supervised machine learning ' \
                             'method. Please provide a mapping of the Gating tree to proceed (see documentation for ' \
                             'help)'
        vprint(f'------ Gating control {ctrl_id} ------')
        for pop_name in progress_bar(self.populations.keys(), verbose):
            if self.search_ctrl_cache(pop_name, ctrl_id) is None:
                self._predict_ctrl_population(pop_name, ctrl_id, model, mappings=tree_map.get(pop_name))

    @staticmethod
    def _check_class_args(klass,
                          method: str,
                          **kwargs) -> bool:
        """
        Check parameters meet class requirements

        Parameters
        ----------
        klass :
            Valid gating class
        method :
            Name of class method to be called
        kwargs :
            Keyword arguments supplied by user

        Returns
        -------
        bool
            True if valid, else False
        """
        try:
            if not inspect.getmro(klass):
                raise GateError(f'{klass} Invalid: must inherit from Gate class')
            klass_args = get_params(klass, required_only=True, exclude_kwargs=True)
            for arg in klass_args:
                if arg == 'data':
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

    def merge(self,
              population_left: str,
              population_right: str,
              new_population_name: str):
        """
        Merge two populations from the same parent, generating a new population saved to population tree

        Parameters
        ----------
        population_left: str :
            Name of left population to merge
        population_right: str :
            Name of right population to merge
        new_population_name: str :
            Name of the new population

        Returns
        -------
        None
        """
        assert new_population_name not in self.populations.keys(), f'{new_population_name} already exists!'
        assert population_left in self.populations.keys(), f'{population_left} not recognised!'
        assert population_right in self.populations.keys(), f'{population_right} not recognised!'
        population_left_parent = self.populations[population_left].parent.name
        population_right_parent = self.populations[population_right].parent.name
        assert population_left_parent == population_right_parent, 'Population parent must match for merging ' \
                                                                  'populations ' \
                                                                  f'left parent = {population_left_parent}, ' \
                                                                  f'right parent = {population_right_parent}'
        parent = self.populations[population_left_parent]
        x, y = self.populations[population_left].geom['x'], self.populations[population_left].geom['y']
        left_idx, right_idx = self.populations[population_left].index, self.populations[population_right].index
        index = np.unique(np.concatenate((left_idx, right_idx)))
        new_population = ChildPopulationCollection(gate_type='merge')
        new_population.add_population(new_population_name)
        parent_df = self.get_population_df(parent.name)
        d = parent_df.loc[index][[x, y]]
        hull = ConvexHull(d)
        polygon = Polygon([(d.values[v, 0], d.values[v, 1]) for v in hull.vertices])
        cords = dict(x=polygon.exterior.xy[0], y=polygon.exterior.xy[1])
        new_population.populations[new_population_name].update_geom(x=x, y=y, shape='poly', cords=cords)
        new_population.populations[new_population_name].update_index(index)
        self.update_populations(output=new_population,
                                parent_name=parent.name,
                                warnings=[])
        name = f'merge_{population_left}_{population_right}'
        kwargs = [('left', population_left), ('right', population_right), ('name', new_population_name),
                  ('x', x), ('y', y), ('child_populations', new_population)]
        new_gate = DataGate(gate_name=name, children=list(new_population.populations.keys()), parent=parent.name,
                            method='merge', kwargs=kwargs, class_='merge')
        self.gates[name] = new_gate

    def subtraction(self,
                    target: list,
                    parent: str,
                    new_population_name: str) -> None:
        """
        Given a target population(s) and a parent population, generate a new population by subtraction of the
        target population from the parent

        Parameters
        ----------
        target: list :
            One or more population names
        parent: str :
            Name of parent population
        new_population_name: str :
            Name of new population generated after subtraction

        Returns
        -------
        None
        """
        assert parent in self.populations.keys(), 'Error: parent population not recognised'
        assert type(target) == list, 'Target should be a list (if only one population, ' \
                                     'should be a list with one element)'
        assert all([t in self.populations.keys() for t in target]), 'Error: target population not recognised'
        assert new_population_name not in self.populations.keys(), f'Error: a population with name ' \
                                                                   f'{new_population_name} already exists'

        x = self.populations[parent].geom['x']
        y = self.populations[parent].geom['y']
        pindex = self.populations[parent].index
        tindex = np.unique(np.concatenate([self.populations[t].index for t in target], axis=0))
        index = np.setdiff1d(pindex, tindex)
        # index = [p for p in pindex if p not in tindex]
        new_population = ChildPopulationCollection(gate_type='sub')
        new_population.add_population(new_population_name)
        new_population.populations[new_population_name].update_geom(x=x, y=y, shape='sub')
        new_population.populations[new_population_name].update_index(index)
        self.update_populations(output=new_population,
                                parent_name=parent,
                                warnings=[])
        kwargs = [('parent', parent), ('target', target), ('name', new_population_name),
                  ('x', x), ('y', y), ('child_populations', new_population)]
        new_gate = DataGate(gate_name=f'{parent}_minus_{target}', children=[new_population_name],
                            parent=parent, method='subtraction', kwargs=kwargs, class_='subtraction')
        self.gates[f'{parent}_minus_{target}'] = new_gate

    def create_gate(self,
                    gate_name: str,
                    parent: str,
                    class_: str,
                    method: str,
                    kwargs: dict,
                    child_populations: ChildPopulationCollection) -> bool:
        """Define a new gate to be used using 'apply' method

        Parameters
        ----------
        gate_name : str
            Name of the gate
        parent : str
            Name of parent population gate is applied to
        class_ : str
            Name of a valid gating class
        method : str
            Name of the class method to invoke upon gating
        kwargs : dict
            Keyword arguments (include constructor arguments and method arguments)
        child_populations : ChildPopulationCollection
            A valid ChildPopulationCollection object describing the resulting populations

        Returns
        -------
        bool
            True if successful, else False

        """
        if gate_name in self.gates.keys():
            print(f'Error: gate with name {gate_name} already exists.')
            return False
        if class_ not in self.gating_classes:
            print(f'Error: invalid gate class, must be one of {self.gating_classes}')
            return False
        kwargs['child_populations'] = child_populations
        if not self._check_class_args(self.gating_classes[class_], method, **kwargs):
            return False
        kwargs = [(k, v) for k, v in kwargs.items()]
        new_gate = DataGate(gate_name=gate_name, children=list(child_populations.populations.keys()), parent=parent,
                            method=method, kwargs=kwargs, class_=class_)
        self.gates[gate_name] = new_gate
        return True

    def _apply_checks(self, gate_name: str) -> DataGate or None:
        """
        Default checks applied whenever a gate is applied

        Parameters
        ----------
        gate_name : str
            Name of gate to apply

        Returns
        -------
        Gate or None
            Gate document (None if checks fail)

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

    def _construct_class_and_gate(self,
                                  gatedoc: DataGate,
                                  kwargs: dict,
                                  feedback: bool = True):
        """Construct a gating class object and apply the intended method for gating

        Parameters
        ----------
        gatedoc : Gate
            Gate document generated with `create_gate`
        kwargs : dict
            keyword arguments for constructor and method
        feedback: bool, (Default value = True):
            If True, feedback printed to stdout

        Returns
        -------
        None
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
        self.update_populations(output,
                                warnings=analyst.warnings,
                                parent_name=gatedoc.parent)
        if feedback:
            for pop in output.populations.keys():
                print(f'New population: {pop}')
                print(f'...proportion of total events: {self.populations[pop].prop_of_total:.3f}')
                print(f'...proportion of parent: {self.populations[pop].prop_of_parent:.3f}')
            print('-----------------------')

    def apply(self,
              gate_name: str,
              plot_output: bool = True,
              feedback: bool = True,
              **kwargs) -> None:
        """Apply a gate to events data (must be generated with `create_gate` first)

        Parameters
        ----------
        gate_name : str
            Name of the gate to apply
        plot_output : bool, (default=True)
            If True, resulting gates will be printed to screen
        feedback : bool, (default=True)
            If True, print feedback
        **kwargs :
            keyword arguments to pass to call to gating method

        Returns
        -------
        None
        """
        gatedoc = self._apply_checks(gate_name)
        if gatedoc is None:
            return None
        gkwargs = {k: v for k, v in gatedoc.kwargs}
        # Add kwargs if given
        for k, v in kwargs.items():
            gkwargs[k] = v
        for fmo in ['fmo_x', 'fmo_y']:
            if fmo in gkwargs.keys():
                try:
                    gkwargs[fmo] = self.search_ctrl_cache(target_population=gatedoc.parent,
                                                          ctrl_id=gkwargs[fmo],
                                                          return_dataframe=True)
                except ValueError:
                    print(f'No control data found for {gkwargs[fmo]}, calling "control_gating"')
                    self.control_gating(ctrl_id=gkwargs[fmo])
                    gkwargs[fmo] = self.search_ctrl_cache(target_population=gatedoc.parent,
                                                          ctrl_id=gkwargs[fmo],
                                                          return_dataframe=True)

        if gatedoc.class_ == 'merge':
            self.merge(population_left=gkwargs.get('left'), population_right=gkwargs.get('right'),
                       new_population_name=gkwargs.get('name'))
        elif gatedoc.class_ == 'subtraction':
            self.subtraction(target=gkwargs.get('target'), parent=gkwargs.get('parent'),
                             new_population_name=gkwargs.get('name'))
        else:
            self._construct_class_and_gate(gatedoc, gkwargs, feedback)
        if plot_output:
            self.plotting.plot_gate(gate_name=gate_name)

    def update_populations(self,
                           output: ChildPopulationCollection,
                           warnings: list,
                           parent_name: str) -> ChildPopulationCollection:
        """Given some ChildPopulationCollection object generated from a gate, update saved populations

        Parameters
        ----------
        output : ChildPopulationCollection
            ChildPopulationCollection object generated from a gate
        warnings : list
            list of warnings generated from gate
        parent_name : str
            name of the parent population

        Returns
        -------
        ChildPopulationCollection
            output
        """
        parent_df = self.get_population_df(parent_name)
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
                                          parent=self.populations[parent_name],
                                          control_idx=dict())
        return output

    def apply_many(self,
                   gates: list = None,
                   apply_all: bool = False,
                   plot_outcome: bool = False,
                   feedback: bool = True) -> None:
        """Apply multiple existing gates sequentially

        Parameters
        ----------
        gates : list, optional
            Name of gates to apply (NOTE: Gates must be provided in sequential order!)
        apply_all : bool (default=False)
            If True, gates is ignored and all current gates will be applied
            (population tree must be empty)
        plot_outcome : bool, (default=False)
            If True, resulting gates will be printed to screen
        feedback : bool, (default=True)
            If True, feedback will be printed to stdout

        Returns
        -------
        None
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
            self.apply(gate_name, plot_output=plot_outcome, feedback=feedback)
        if feedback:
            print('Complete!')

    @staticmethod
    def _geom_bool(geom: dict,
                   definition: str,
                   parent: pd.DataFrame):
        """
        Internal function. Edit a population defined by two thresholds (in x-axis and y-axis plane).
        Given some geom dictionary and a parent data-frame, parse the thresholds stored in the geom
        and return index of parent data-frame when filtered according to thresholds

        Parameters
        ----------
        geom: dict
            Dictionary of geometric description of gate. Expected to contain keys: x, y, threshold_x, threshold_y
        definition : str
            Definition associated to population that is being edited (e.g. '++' relates to a population that is
            positive in both x and y dimension
        parent : Pandas.DataFrame
            Parent DataFrame that relates to the parent of the population being edited

        Returns
        -------
        Numpy.array
            Array of index values for new edited population
        """
        parent = parent.round(decimals=2)
        x_, y_ = geom['x'], geom['y']
        tx, ty = round(geom['threshold_x'], 2), round(geom['threshold_y'], 2)
        if definition == '++':
            return parent[(parent[x_] > tx) & (parent[y_] > ty)].index.values
        if definition == '--':
            return parent[(parent[x_] < tx) & (parent[y_] < ty)].index.values
        if definition == '+-':
            return parent[(parent[x_] > tx) & (parent[y_] < ty)].index.values
        if definition == '-+':
            return parent[(parent[x_] < tx) & (parent[y_] > ty)].index.values
        raise ValueError('Definition must have a value of "+-", "-+", "--", or "++" for a 2D threshold gate')

    @staticmethod
    def _update_threshold_1d(geom: dict,
                             parent: pd.DataFrame):
        """
        Internal function. Edit a population defined by a threshold (in x-axis plane).
        Given some geom dictionary and a parent data-frame, parse the thresholds stored in the geom
        and return index of parent data-frame when filtered according to thresholds

        Parameters
        ----------
        geom: dict
            Dictionary of geometric description of gate. Expected to contain keys: x, threshold, transform_x
        parent : Pandas.DataFrame
            Parent DataFrame that relates to the parent of the population being edited

        Returns
        -------
        Numpy.array
            Array of index values for new edited population
        """
        assert 'threshold' in geom.keys(), 'Geom must contain a key "threshold" with a float value'
        assert 'transform_x' in geom.keys(), 'Geom must contain a key "transform_x", ' \
                                             'the transform method for the x-axis'
        assert 'definition' in geom.keys(), 'Geom must contain key "definition", a string value that indicates ' \
                                            'if population is the "positive" or "negative"'
        if geom['definition'] == '+':
            return parent[parent[geom['x']] >= geom['threshold']].index.values
        if geom['definition'] == '-':
            return parent[parent[geom['x']] < geom['threshold']].index.values
        raise ValueError('Definition must have a value of "+" or "-" for a 1D threshold gate')

    @staticmethod
    def _update_rect(geom: dict,
                     parent: pd.DataFrame):
        """
        Internal function. Edit a population defined by a rectangular gate.
        Given some geom dictionary and a parent data-frame, parse the rectangular gate parameters stored in the geom
        and return index of parent data-frame when filtered according to geom

        Parameters
        ----------
        geom: dict
            Dictionary of geometric description of gate. Expected to contain keys: x, y, transform_x, transform_y,
            x_min, x_max, y_min, y_max
        parent : Pandas.DataFrame
            Parent DataFrame that relates to the parent of the population being edited

        Returns
        -------
        Numpy.array
            Array of index values for new edited population
        """
        keys = ['x_min', 'x_max', 'y_min', 'y_max']
        assert 'definition' in geom.keys(), 'Geom must contain key "definition", a string value that indicates ' \
                                            'if population is the "positive" or "negative"'
        assert all([r in geom.keys() for r in keys]), f'Geom must contain keys {keys} both with a float value'
        assert all(
            [t in geom.keys() for t in ['transform_x', 'transform_y']]), 'Geom must contain a key "transform_x", ' \
                                                                         'the transform method for the x-axis AND ' \
                                                                         'a key "transform_y", ' \
                                                                         'the transform method for the y-axis'
        assert geom.get('y'), 'Geom is missing value for "y"'

        x = (parent[geom['x']] >= geom['x_min']) & (parent[geom['x']] <= geom['x_max'])
        y = (parent[geom['y']] >= geom['y_min']) & (parent[geom['y']] <= geom['y_max'])
        pos = parent[x & y]
        if geom['definition'] == '+':
            return pos.index.values
        if geom['definition'] == '-':
            return parent[~parent.index.isin(pos.index)].index.values
        raise ValueError('Definition must have a value of "+" or "-" for a rectangular geom')

    @staticmethod
    def _update_ellipse(geom: dict,
                        parent: pd.DataFrame):
        keys = ['centroid', 'width', 'height', 'angle']
        assert 'definition' in geom.keys(), 'Geom must contain key "definition", a string value that indicates ' \
                                            'if population is the "positive" or "negative"'
        assert all([c in geom.keys() for c in
                    keys]), f'Geom must contain keys {keys}; note, centroid must be a tuple and all others a float value'
        assert geom.get('y'), 'Geom is missing value for "y"'
        assert all(
            [t in geom.keys() for t in ['transform_x', 'transform_y']]), 'Geom must contain a key "transform_x", ' \
                                                                         'the transform method for the x-axis AND ' \
                                                                         'a key "transform_y", ' \
                                                                         'the transform method for the y-axis'

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
        raise ValueError('Definition must have a value of "+" or "-" for a ellipse geom')##

    @staticmethod
    def _update_poly(geom: dict,
                     x: str,
                     y: str,
                     parent: pd.DataFrame):
        """
         Internal function. Edit a population defined by a polygon gate.
         Given some geom dictionary and a parent data-frame, parse the polygon gate parameters stored in the geom
         and return index of parent data-frame when filtered according to geom

         Parameters
         ----------
         geom: dict
             Dictionary of geometric description of gate. Expected to contain keys: x, y, transform_x, transform_y,
             cords
         parent : Pandas.DataFrame
             Parent DataFrame that relates to the parent of the population being edited

         Returns
         -------
         Numpy.array
             Array of index values for new edited population
         """
        keys = ['cords', 'transform_x', 'transform_y', 'x', 'y']
        assert all([c in geom.keys() for c in keys]), f'Geom must contain keys {keys}'
        assert type(geom.get('cords')) == dict, 'Cords should be of type dictionary with keys: x, y'
        cords = geom.get('cords')
        assert all([_ in cords.keys() for _ in ['x', 'y']]), 'Cords should contain keys: x, y'

        poly = Polygon([(x, y) for x, y in zip(cords['x'], cords['y'])])
        pos = inside_polygon(parent, x, y, poly)
        return pos.index

    def _update_threshold_2d(self,
                             geom: dict,
                             parent: pd.DataFrame):
        """
        Internal function. Edit a population defined by a two thresholds (in x-axis and y-axis plane).
        Given some geom dictionary and a parent data-frame, parse the thresholds stored in the geom
        and return index of parent data-frame when filtered according to thresholds

        Parameters
        ----------
        geom: dict
            Dictionary of geometric description of gate. Expected to contain keys: x, y, threshold_x, threshold_y,
            transform_x, transform_y
        parent : Pandas.DataFrame
            Parent DataFrame that relates to the parent of the population being edited

        Returns
        -------
        Numpy.array
            Array of index values for new edited population
        """
        assert 'definition' in geom.keys(), 'Geom must contain key "definition", a string value that indicates ' \
                                            'if population is the "positive" or "negative"'
        assert all([t in geom.keys() for t in ['threshold_x', 'threshold_y']]), \
            'Geom must contain keys "threshold_x" and "threshold_y" both with a float value'
        assert geom.get('y'), 'Geom is missing value for "y"'
        assert all(
            [t in geom.keys() for t in ['transform_x', 'transform_y']]), 'Geom must contain a key "transform_x", ' \
                                                                         'the transform method for the x-axis AND ' \
                                                                         'a key "transform_y", ' \
                                                                         'the transform method for the y-axis'

        if type(geom['definition']) == list:
            idx = list(map(lambda d: self._geom_bool(geom, d, parent), geom['definition']))
            return [i for l in idx for i in l]
        else:
            return self._geom_bool(geom, geom['definition'], parent)

    def _update_index(self,
                      population_name: str,
                      geom: dict):
        """Given some new gating geom and the name of a population to update, update the population index

        Parameters
        ----------
        population_name : str
            name of population to update
        geom : dict
            valid dictionary describing geom

        Returns
        -------
        Numpy.array

        """
        assert population_name in self.populations.keys(), f'Population {population_name} does not exist'
        assert 'shape' in geom, 'Geom missing key argument "shape"'
        parent_name = self.populations[population_name].parent.name
        parent = self.get_population_df(parent_name, transform=False)
        transform_x, transform_y = geom.get('transform_x'), geom.get('transform_y')
        x, y = geom.get('x'), geom.get('y')
        assert x, 'Geom is missing value for "x"'
        if transform_x is not None and x is not None:
            parent = apply_transform(parent,
                                     transform_method=transform_x,
                                     features_to_transform=[x])
        if transform_y is not None and y is not None:
            parent = apply_transform(parent,
                                     transform_method=transform_y,
                                     features_to_transform=[y])

        if geom['shape'] == 'threshold':
            return self._update_threshold_1d(geom=geom, parent=parent)

        if geom['shape'] == '2d_threshold':
            return self._update_threshold_2d(geom=geom, parent=parent)

        if geom['shape'] == 'rect':
            return self._update_rect(geom=geom, parent=parent)

        if geom['shape'] == 'ellipse':
            return self._update_ellipse(geom=geom, parent=parent)

        if geom['shape'] == 'poly':
            return self._update_poly(geom=geom, x=x, y=y, parent=parent)

        raise ValueError('Geom shape not recognised, should be one of: threshold, 2d_threshold, ellipse, rect, poly')

    def edit_gate(self,
                  gate_name: str,
                  updated_geom: dict,
                  delete: bool = True):
        """Manually replace the outcome of a gate by updating the geom of it's child populations.

        Parameters
        ----------
        gate_name : str
            name of gate to update
        updated_geom : dict
            new geom as valid dictionary
        delete : bool, (default=True)
            if True, all populations downstream of immediate children will be removed. This is recommended
            as edit_gate does not update the index of populations downstream of the immediate children.

        Returns
        -------
        None
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
            self.populations[c].index = self._update_index(c, updated_geom[c])
            effected_populations = effected_populations + self.find_dependencies(population=c)
            immediate_children = immediate_children + [n.name for n in self.populations[c].children]
        effected_gates = [name for name, gate in self.gates.items() if gate.parent in effected_populations]
        print(f'The following gates are downstream of {gate_name} and will need to be applied again: {effected_gates}')
        if delete:
            for c in immediate_children:
                self.remove_population(c)
        print('Edit complete!')

    def nudge_threshold(self,
                        gate_name: str,
                        new_x: float,
                        new_y: float or None = None):
        """
        Given some DensityThreshold gate, update the threshold calculated for child populations

        Parameters
        ----------
        gate_name: str :
            Name of gate to update
        new_x: float :
            New x-axis threshold
        new_y: float, optional
            New y-axis threshold

        Returns
        -------
        None
        """
        assert gate_name in self.gates.keys(), 'Invalid gate name'
        class_, method = self.gates[gate_name].class_.lower(), self.gates[gate_name].method.lower()
        assert 'threshold' in class_ or 'threshold' in method, 'Can only nudge threshold gates'
        children = self.gates[gate_name].children
        geoms = {c: self.fetch_geom(c) for c in children}
        for c in children:
            if self.gates[gate_name].method == 'gate_1d':
                geoms[c]['threshold'] = new_x
            else:
                geoms[c]['threshold_x'] = new_x
                if new_y is not None:
                    geoms[c]['threshold_y'] = new_y
        self.edit_gate(gate_name, updated_geom=geoms)

    def find_dependencies(self,
                          population: str) -> list or None:
        """For a given population find all dependencies

        Parameters
        ----------
        population : str
            population name

        Returns
        -------
        list or None
            List of populations dependent on given population

        """
        if population not in self.populations.keys():
            print(f'Error: population {population} does not exist; '
                  f'valid population names include: {self.populations.keys()}')
            return None
        root = self.populations['root']
        node = self.populations[population]
        return [x.name for x in findall(root, filter_=lambda n: node in n.path)]

    def remove_population(self,
                          population_name: str,
                          hard_delete: bool = False) -> None:
        """
        Remove a population

        Parameters
        ----------
        population_name : str
            name of population to remove
        hard_delete : bool, (default=False)
            if True, population and dependencies will be removed from database

        Returns
        -------
        None

        """
        if population_name not in self.populations.keys():
            print(f'{population_name} does not exist')
            return None
        downstream_populations = self.find_dependencies(population=population_name)
        self.populations[population_name].parent = None
        for x in downstream_populations:
            self.populations.pop(x)
        if hard_delete:
            self.filegroup.delete_populations(downstream_populations)
            self.filegroup = self.filegroup.save()

    def remove_gate(self,
                    gate_name: str,
                    propagate: bool = True) -> list and list or None:
        """
        Remove gate

        Parameters
        ----------
        gate_name : str
            name of gate to remove
        propagate : bool, (default=True)
            If True, downstream gates and effected populations will also be removed

        Returns
        -------
        (list, list) or None
            list of removed gates, list of removed populations

        """
        if gate_name not in self.gates.keys():
            print('Error: invalid gate name')
            return None
        gate = self.gates[gate_name]
        if not gate.children or not propagate:
            self.gates.pop(gate_name)
            return None
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

    def print_population_tree(self,
                              image: bool = False,
                              image_name: str or None = None) -> None:
        """
        Generate a tree diagram of the populations associated to this Gating object and print to stdout

        Parameters
        ----------
        image : bool, (default=False)
            if True, an image will be saved to the working directory
        image_name : str, optional
            name of the resulting image file, ignored if image = False (optional; default name is of
            format `filename_populations.png`

        Returns
        -------
        None

        """
        root = self.populations['root']
        if image:
            if image_name is None:
                image_name = f'{self.id}_population_tree.png'
            DotExporter(root).to_picture(image_name)
        for pre, fill, node in RenderTree(root):
            print('%s%s' % (pre, node.name))

    def _population_to_mongo(self,
                             population_name: str) -> Population:
        """
        Convert a population into a mongoengine Population document

        Parameters
        ----------
        population_name : str
            Name of population to convert

        Returns
        -------
        Population
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
                               geom=geom,
                               n=len(pop_node.index))
        pop_mongo.save_index(pop_node.index)
        if pop_node.control_idx:
            pop_mongo.save_control_idx(pop_node.control_idx)
        return pop_mongo

    def _save_ctrl_idx(self):
        """
        Saves the cached control index of each current population to the database

        Returns
        -------
        None
        """
        updated_populations = list()
        for pop in self.filegroup.populations:
            ctrl_idx = self.populations[pop.population_name].control_idx
            if not ctrl_idx:
                updated_populations.append(pop)
                continue
            controls = {k: ControlIndex(control_id=k) for k in ctrl_idx.keys()}
            for ctrl_id, idx in ctrl_idx.items():
                controls[ctrl_id].save_index(idx)
            pop.control_idx = list(controls.values())
            updated_populations.append(pop)
        self.filegroup.populations = updated_populations
        self.filegroup.save()

    def save(self,
             overwrite: bool = False,
             feedback: bool = True) -> bool:
        """
        Save all gates and population's to mongoDB

        Parameters
        ----------
        overwrite : bool, (default=False)
            If True, existing populations/gates for sample will be overwritten
        feedback: bool, (default=True)
             If True, feedback printed to stdout

        Returns
        -------
        bool
            True if successful else False

        """
        existing_pops = list(self.filegroup.list_populations())

        # Update populations
        populations_to_save = list()
        for name in self.populations.keys():
            if name in existing_pops:
                existing_population = FileGroup.objects(id=self.mongo_id).get().get_population(name)
                if np.array_equal(existing_population.load_index(), self.populations.get(name).index):
                    populations_to_save.append(existing_population)
                    continue
                if not overwrite:
                    raise ValueError(f'The index for population {name} has been changed, change "overwrite" to '
                                     f'True to overwrite existing data; note this will delete any clusters '
                                     f'currently associated to this population')
                else:
                    if existing_population.clustering:
                        print(f'Warning: index for {name} has changed and the associated clusters '
                              f'are now invalid')
                    populations_to_save.append(self._population_to_mongo(name))
            else:
                populations_to_save.append(self._population_to_mongo(name))
        self.filegroup.populations = populations_to_save

        # Update gates
        self.filegroup.gates = [self._serailise_gate(gate) for gate in self.gates.values()]
        self.filegroup = self.filegroup.save()
        self._save_ctrl_idx()
        if feedback:
            print('Saved successfully!')
        return True

    def _cluster_idx(self,
                     cluster_id: str,
                     clustering_root: str,
                     meta: bool = True):
        """Fetch the index of a given cluster/meta-cluster in associated sample

        Parameters
        ----------
        cluster_id : str
            name of cluster if interest
        clustering_root : str
            name of root population for cluster of interest
        meta : bool, (default=True)
            if True, search for a meta-cluster if False, treat cluster_id as unique clustering ID

        Returns
        -------
        Numpy.array
            numpy array for index of events contained in cluster

        """
        assert clustering_root in self.populations.keys(), f'Invalid root name, must be one of {self.populations.keys()}'
        fg = FileGroup.objects(id=self.mongo_id).get()
        croot_pop = [p for p in fg.populations if p.population_name == clustering_root][0]
        _, idx = croot_pop.get_cluster(cluster_id=cluster_id, meta=meta)
        return idx

    def register_as_invalid(self):
        """
        Flags the sample associated to this Gating instance as invalid and saves state to database
        """
        fg = FileGroup.objects(id=self.mongo_id).get()
        if fg.flags:
            fg.flags = fg.flags + ',invalid'
        else:
            fg.flags = 'invalid'
        fg.save()

    def check_downstream_overlaps(self,
                                  root_population: str,
                                  population_labels: list) -> bool:
        """
        Check if a chosen root population is downstream of target populations (population_labels).

        Parameters
        ----------
        root_population: str
            Name of the root population (presumed parent)
        population_labels: list
            List of target populations. Each population in this list will be checked, asserting that
            the population is downstream of the root_population

        Returns
        -------
        bool
            True if one or more populations has the root population downstream of itself, else False
        """
        downstream_overlaps = False
        for pop_i in population_labels:
            dependencies = self.find_dependencies(pop_i)
            if root_population in dependencies:
                print(f'Error: population {pop_i} is upstream from the chosen root population {root_population}')
                downstream_overlaps = True
            for pop_j in population_labels:
                if pop_j == pop_i:
                    continue
                if pop_j in dependencies:
                    print(f'Error: population {pop_j} is a dependency of population {pop_i} (i.e. it is downstream '
                          f'from this population). This will result in invalid labelling. If you wish to continue '
                          f'with these population targets, please set multi_label parameter to True')
                    downstream_overlaps = True
        return downstream_overlaps


class Template(Gating):
    """Generate a reusable template for gating. Inherits all functionality of Gating class."""
    def save_new_template(self,
                          template_name: str,
                          overwrite: bool = True) -> bool:
        """Save template structure as a GatingStrategy

        Parameters
        ----------
        template_name : str
            name of the template
        overwrite : bool, (default=True)
            If True, any existing template with the same name will be overwritten

        Returns
        --------
        bool
            True if successful, else False

        """
        gating_strategy = GatingStrategy.objects(template_name=template_name)
        if gating_strategy:
            if not overwrite:
                print(f'Template with name {template_name} already exists, set parameter '
                      f'`overwrite` to True to continue')
                return False
            print(f'Overwriting existing gating template {template_name}')
            gating_strategy = gating_strategy[0]
            gating_strategy.gates = [self._serailise_gate(gate) for gate in list(self.gates.values())]
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
            gating_strategy.gates = [self._serailise_gate(gate) for gate in list(self.gates.values())]
            gating_strategy.save()
            self.experiment.gating_templates.append(gating_strategy)
            self.experiment.save()
            return True

    def load_template(self,
                      template_name: str) -> bool:
        """Load gates from a template GatingStrategy

        Parameters
        ----------
        template_name : str
            name of template to load

        Returns
        -------
        bool
            True if successful, else False

        """
        gating_strategy = GatingStrategy.objects(template_name=template_name)
        if gating_strategy:
            for gate in gating_strategy[0].gates:
                self._deserialise_gate(gate)
            return True
        else:
            print(f'No template with name {template_name}')
            return False