from ..data.experiments import Experiment
from ..data.populations import Polygon, Threshold, Population, merge_populations
from ..data.gates import Gate, PreProcess, PostProcess
from ..data.gating_strategy import GatingStrategy, Action
from .transforms import apply_transform
from ..utilities import inside_polygon
from ..feedback import vprint
from .plotting import CreatePlot
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn import cluster, mixture
from mongoengine.errors import DoesNotExist
from anytree.exporter import DotExporter
from anytree import Node, RenderTree
from anytree.search import findall
from warnings import warn
from typing import List
import pandas as pd
import numpy as np
import inspect


def valid_sklearn(klass: str):
    """
    Given the name of a Scikit-Learn class, checks validity. If invalid, raises Assertion error,
    otherwise returns the class name.

    Parameters
    ----------
    klass: str

    Returns
    -------
    str
    """
    valid_clusters = [x[0] for x in inspect.getmembers(cluster, inspect.isclass)
                      if 'sklearn.cluster' in x[1].__module__]
    valid_mixtures = [x[0] for x in inspect.getmembers(mixture, inspect.isclass)
                      if 'sklearn.mixture' in x[1].__module__]
    valid = valid_clusters + valid_mixtures + ["HDBSCAN"]
    err = f"""Invalid class name. Must be one of the following from Scikit-Learn's cluster module: {valid_clusters};
 or from Scikit-Learn's mixture module: {valid_mixtures}; or 'HDBSCAN'"""
    assert klass in valid, err
    return klass


def _edit_threshold_idx(parent: pd.DataFrame,
                        definition: str,
                        new_geom: Threshold):
    if definition == "+":
        return parent[parent[new_geom.x] > new_geom.x_threshold]
    if definition == "-":
        return parent[parent[new_geom.x] <= new_geom.x_threshold]
    if definition == "--":
        return parent[(parent[new_geom.x] <= new_geom.x_threshold) &
                      (parent[new_geom.y] <= new_geom.y_threshold)].index.values
    if definition == "++":
        return parent[(parent[new_geom.x] > new_geom.x_threshold) &
                      (parent[new_geom.y] > new_geom.y_threshold)].index.values
    if definition == "+-":
        return parent[(parent[new_geom.x] > new_geom.x_threshold) &
                      (parent[new_geom.y] <= new_geom.y_threshold)].index.values
    if definition == "-+":
        return parent[(parent[new_geom.x] <= new_geom.x_threshold) &
                      (parent[new_geom.y] > new_geom.y_threshold)].index.values
    raise ValueError("Invalid definition, cannot edit gate")


def _gate_feature_check(x: str,
                        y: str or None,
                        valid_columns: list,
                        preprocessing_kwargs: dict):
    features_to_check = [i for i in [x, y] if i is not None]
    if any(i not in valid_columns for i in features_to_check):
        if not preprocessing_kwargs.get("dim_reduction"):
            err = f"x or y are invalid values are invalid; valid column names as: {valid_columns}"
            raise ValueError(err)
        else:
            assert x == "embedding1", "If using dim_reduction, x should have a value 'embedding1'"
            assert y == "embedding2", "If using dim_reduction, y should have a value 'embedding2'"


def _gate_validate_shape(shape: str,
                         method: str,
                         method_kwargs: dict,
                         preprocessing_kwargs: dict):
    if shape == "threshold":
        if method not in ["DensityGate", "ManualGate"]:
            warn("Shape set to 'threshold', defaulting to DensityGate")
            method = "DensityGate"
    elif shape == "ellipse":
        if method is None:
            warn("Method not given, defaulting to BayesianGaussianMixture")
            method = "BayesianGaussianMixture"
            method_kwargs["n_components"] = 5
        else:
            err = "For an elliptical gate, expect method 'GaussianMixture', 'BayesianGaussianMixture', " \
                  "or 'MiniBatchKMeans'"
            valid = ["GaussianMixture", "BayesianGaussianMixture", "MiniBatchKMeans"]
            assert method in valid, err
    elif shape == "polygon":
        if not method:
            warn("No method specified for Polygon Gate, defaulting to MiniBatchKMeans")
            method = "MiniBatchKMeans"
            method_kwargs["n"] = 5
        elif method != "ManualGate":
            method = valid_sklearn(method)
            if "dbscan" in method.lower():
                if preprocessing_kwargs.get("downsample_method") is None:
                    warn("DBSCAN and HDBSCAN do not scale well and it is recommended that downsampling is performed")
    return method, method_kwargs


class Gating:
    """
    Central class for performing semi-automated gating and storing gating information on an FCS FileGroup
    of a single sample.

    Parameters
    -----------
    experiment: FCSExperiment
        experiment you're currently working on
    sample_id: str
        name of the sample to analyse (must belong to experiment)
    gating_strategy: str (optional)
        Name of the gating strategy to load along with associated gates. If not provided, will attempt to
        locate gating strategy currently associated to the chosen sample. If there is no gating strategy associated
        to the current sample and you wish to create a new gating strategy, provide a unique name and after creating
        the gates, call `save_gating_strategy`
    include_controls: bool, (default=True)
        If True and FMOs are included for specified samples, the FMO data will also be loaded into the Gating object
    verbose: bool (default=True)
        Whether to provide feedback
    gate_ctrls_adhoc: bool (default=True)
        If True, index for control samples will be estimated automatically whenever a gate is applied
    ctrl_gate_cv: int (default=10)
        Number of folds to use in cross-validation when estimating index for control samples
    """

    def __init__(self,
                 experiment: Experiment,
                 sample_id: str,
                 gating_strategy: str or None = None,
                 include_controls=True,
                 verbose: bool = True,
                 gate_ctrls_adhoc: bool = True,
                 ctrl_gate_cv: int = 10):
        self.data = experiment.get_data(sample_id=sample_id, sample_size=None, include_controls=include_controls)
        self.id = sample_id
        self.mongo_id = experiment.get_sample_mid(sample_id)
        self.experiment = experiment
        self.filegroup = experiment.get_sample(sample_id)
        self.populations = dict()
        self.tree = self._construct_tree()
        self.verbose = verbose
        self.vprint = vprint(verbose)
        self.crtl_gate_ad_hoc = gate_ctrls_adhoc
        self.ctrl_gate_cv = ctrl_gate_cv

        if gating_strategy is None:
            assert self.filegroup.gating_strategy, f"{sample_id} has not been previously 'gated', please provide the name to " \
                                                   f"an existing gating strategy to be applied or provide a new name if you wish " \
                                                   f"to create a new gating strategy."
            self.template = self.filegroup.gating_strategy
        else:
            try:
                self.template = GatingStrategy.objects(template_name=gating_strategy).get()
            except DoesNotExist:
                self.template = GatingStrategy(template_name=gating_strategy)
        self.gates = {g.gate_name: g for g in self.template.gates}
        self.actions = {x.action_name: x for x in self.template.actions}

    def _construct_tree(self):
        """
        Construct the population tree; that is, a dictionary of Population objects and AnyTree nodes.

        Returns
        -------
        dict
        """
        if not self.filegroup.populations:
            # No population currently exist for this FileGroup. Init with root population
            self.populations = {"root": Population(population_name="root",
                                                   index=self.data.get("primary").index.values,
                                                   parent="root",
                                                   n=len(self.data.get("primary").index.values))}
            if "controls" in self.data.keys():
                for ctrl_id, ctrl_data in self.data.get("controls").items():
                    self.populations["root"].set_ctrl_index(**{ctrl_id: ctrl_data.index.values})
            return {"root": Node(name="root", parent=None)}
        assert "root" in [p.population_name for p in self.filegroup.populations], \
            "Invalid FileGroup, must contain 'root' population"
        self.populations = {p.population_name: p for p in self.filegroup.populations}
        tree = {"root": Node(name="root", parent=None)}
        database_populations = [p for p in self.filegroup.populations if p.population_name != 'root']
        i = 0
        while len(database_populations) > 0:
            if i >= len(database_populations):
                # Loop back around
                i = 0
            branch = self._construct_branch(tree, database_populations[i])
            if branch is not None:
                tree = branch
                database_populations = [p for p in database_populations
                                        if p.population_name != database_populations[i].population_name]
            else:
                i = i + 1
        return tree

    @staticmethod
    def _construct_branch(tree: dict,
                          new_population: Population):
        """
        Construct a single branch of a population tree (see `construct_tree`)

        Parameters
        ----------
        tree: dict
        new_population: Population

        Returns
        -------
        dict
        """
        if new_population.parent not in tree.keys():
            return None
        tree[new_population.population_name] = Node(name=new_population.population_name,
                                                    parent=tree[new_population.parent])
        return tree

    def save_sample(self,
                    overwrite: bool = False):
        """
        Save the actions applied using this Gating object to the
        sample.

        Parameters
        ----------
        overwrite: bool (default=False)
            Overwrite existing populations?

        Returns
        -------
        None
        """
        if self.filegroup.populations:
            assert overwrite, f"{self.id} has previously been gated and has existing populations. To overwrite " \
                              f"this data set 'overwrite' to True"
        self.filegroup.populations = list(self.populations.values())
        self.filegroup.gating_strategy = self.template
        self.filegroup.save()

    def save_gating_strategy(self,
                             overwrite: bool = False):
        if len(GatingStrategy.objects(template_name=self.template.template_name)) > 0:
            assert overwrite, f"Template with name {self.template.template_name} already exists. " \
                              f"Set 'overwrite' to True to overwrite existing template"
            template_name = self.template.template_name
            self.template.delete()
            self.template = GatingStrategy(template_name=template_name)
        self.template.gates = list(self.gates.values())
        self.template.save()

    def clear_gates(self):
        """Remove all currently associated gates."""
        self.gates = dict()

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

    def get_population_df(self,
                          population_name: str,
                          transform: str or None = 'logicle',
                          transform_features: list or str or dict = 'all',
                          ctrl_id: str or None = None) -> pd.DataFrame or None:
        """
        Retrieve a population as a pandas dataframe

        Parameters
        ----------
        population_name : str
            name of population to retrieve
        transform : str or None, (default='logicle')
            transformation method to apply, default = 'logicle' (ignored if transform is False)
        transform_features : list or str, (default='all')
            argument specifying which columns to transform in the returned dataframe. Can either
            be a string value of 'all' (transform all columns), 'fluorochromes' (transform all columns corresponding to a
            fluorochrome) or a list of valid column names
        ctrl_id: str, optional
            If given, retrieves DataFrame of data from control file rather than primary data

        Returns
        -------
        Pandas.DataFrame or None
            Population DataFrame

        """
        assert population_name in self.populations.keys(), f'Population {population_name} not recognised'
        if ctrl_id is None:
            idx = self.populations[population_name].index
            data = self.data.get("primary").loc[idx]
        else:
            idx = self.populations[population_name].ctrl_index.get(ctrl_id)
            if idx is None:
                self.control_gate(population=self.populations.get(population_name), ctrl_id=ctrl_id)
            idx = self.populations[population_name].ctrl_index.get(ctrl_id)
            data = self.data.get("controls").get(ctrl_id).loc[idx]
        if transform is None:
            return data
        return apply_transform(data,
                               features_to_transform=transform_features,
                               transform_method=transform)

    def get_labelled_population_df(self,
                                   population_name: str,
                                   transform: str or None = 'logicle',
                                   transform_features: list or str = 'all'):
        data = self.get_population_df(population_name,
                                      transform,
                                      transform_features)
        data['label'] = None
        dependencies = self.list_downstream_populations(population_name)
        for pop in dependencies:
            idx = self.populations[pop].index
            data.loc[idx, 'label'] = pop
        return data

    def list_downstream_populations(self,
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
        assert population in self.populations.keys(), f'population {population} does not exist; ' \
                                                      f'valid population names include: {self.populations.keys()}'
        root = self.tree['root']
        node = self.tree[population]
        dependencies = [x.name for x in findall(root, filter_=lambda n: node in n.path)]
        return [p for p in dependencies if p != population]

    def list_dependencies(self,
                          population: str) -> list:
        """
        For given population list all populations that this population depends on (upstream in the same branch)

        Parameters
        ----------
        population

        Returns
        -------
        list
        """
        assert population in self.populations.keys(), f"population {population} does not exist"
        root = self.populations['root']
        node = self.populations[population]
        return [x.name for x in findall(root, filter_=lambda n: node in n.path) if x.name != population]

    def list_child_populations(self,
                               population: str):
        assert population in self.populations.keys(), f'population {population} does not exist; ' \
                                                      f'valid population names include: {self.populations.keys()}'
        return [x.name for x in self.tree.get(population).children]

    def valid_populations(self,
                          populations: list):
        """
        Given a list of populations, check validity and return list of valid populations

        Parameters
        ----------
        populations : list
            list of populations to check

        Returns
        -------
        List
            Valid populations
        """
        valid = list()
        for pop in populations:
            if pop not in self.populations.keys():
                self.vprint(f'{pop} is not a valid population')
            else:
                valid.append(pop)
        return valid

    def create_gate(self,
                    gate_name: str,
                    parent: str,
                    x: str,
                    shape: str,
                    y: str or None = None,
                    binary: bool = True,
                    method: str or None = None,
                    method_kwargs: dict or None = None,
                    preprocessing_kwargs: dict or None = None,
                    postprocessing_kwargs: dict or None = None):
        preprocessing_kwargs = preprocessing_kwargs or dict()
        postprocessing_kwargs = postprocessing_kwargs or dict()
        method_kwargs = method_kwargs or dict()
        assert gate_name not in self.gates.keys(), f"{gate_name} already exists!"
        err = """Gate should have one of the following shapes: ["threshold", "polygon", "ellipse"]"""
        assert shape in ["threshold", "polygon", "ellipse"], err
        assert parent in self.populations.keys(), "Invalid parent (does not exist)"
        _gate_feature_check(x=x, y=y, valid_columns=self.data.get("primary").columns, preprocessing_kwargs=preprocessing_kwargs)
        method, method_kwargs = _gate_validate_shape(shape=shape,
                                                     method=method,
                                                     method_kwargs=method_kwargs,
                                                     preprocessing_kwargs=preprocessing_kwargs)
        gate = Gate(gate_name=gate_name,
                    parent=parent,
                    shape=shape,
                    x=x,
                    y=y,
                    binary=binary,
                    method=method,
                    method_kwargs=method_kwargs,
                    preprocessing=PreProcess(**preprocessing_kwargs),
                    postprocessing=PostProcess(**postprocessing_kwargs))
        return gate

    def plot_gate(self,
                  gate: Gate or str,
                  create_plot_kwargs: dict or None = None,
                  gate_plot_kwargs: dict or None = None,
                  populations: list or None = None):
        if isinstance(gate, str):
            assert gate in self.gates.keys(), f"Invalid gate, {gate} does not exist. Must be one of: {self.gates.keys()}"
            gate = self.gates[gate]
        if create_plot_kwargs is None:
            create_plot_kwargs = {}
        if gate_plot_kwargs is None:
            gate_plot_kwargs = {}
        data = self.get_population_df(population_name=gate.parent, transform=None)
        if populations is None:
            children = [c.population_name for c in gate.children]
            assert children, \
                f"{gate.parent} children do not exist in current population tree. Has this gate been applied?"
            populations = [self.populations.get(x) for x in children]
        plotting = CreatePlot(transform_x=gate.preprocessing.transform_x,
                              transform_y=gate.preprocessing.transform_y,
                              xlabel=gate.x,
                              ylabel=gate.y,
                              title=gate.gate_name,
                              **create_plot_kwargs)
        return plotting.plot_gate(gate=gate,
                                  parent=data,
                                  children=populations,
                                  **gate_plot_kwargs)

    def plot_population(self,
                        population: str,
                        x: str,
                        y: str or None = None,
                        create_plot_kwargs: dict or None = None,
                        matplotlib_hist2d_kwargs: dict or None = None):
        create_plot_kwargs = create_plot_kwargs or {}
        matplotlib_hist2d_kwargs = matplotlib_hist2d_kwargs or {}
        data = self.get_population_df(population_name=population, transform=None)
        plotting = CreatePlot(**create_plot_kwargs)
        return plotting.plot(data=data,
                             x=x,
                             y=y,
                             **matplotlib_hist2d_kwargs)

    def plot_backgate(self,
                      parent: str,
                      children: list,
                      x: str,
                      y: str or None = None,
                      create_plot_kwargs: dict or None = None,
                      backgate_kwargs: dict or None = None):
        if create_plot_kwargs is None:
            create_plot_kwargs = {}
        if backgate_kwargs is None:
            backgate_kwargs = {}
        valid_children = self.list_downstream_populations(parent)
        assert all([x in valid_children for x in children]), f"One or more given children is not a valid downstream " \
                                                             f"population of {parent}. Valid downstream populations: {valid_children}"
        parent = self.get_population_df(population_name=parent, transform=None)
        children = {x: self.get_population_df(population_name=x, transform=None) for x in children}
        plotting = CreatePlot(**create_plot_kwargs)
        return plotting.backgate(parent=parent,
                                 children=children,
                                 x=x,
                                 y=y,
                                 **backgate_kwargs)

    def _apply(self,
               gate: Gate):
        data = self.get_population_df(population_name=gate.parent, transform=None)
        ctrl = None
        if gate.ctrl_id:
            ctrl = self.get_population_df(population_name=gate.parent, transform=None, ctrl_id=gate.ctrl_id)
        populations = gate.apply(data=data,
                                 ctrl=ctrl,
                                 verbose=self.verbose)
        return populations

    def preview(self,
                gate: Gate or str,
                stats: bool = True,
                create_plot_kwargs: dict or None = None,
                plot_gate_kwargs: dict or None = None):
        if isinstance(gate, str):
            assert gate in self.gates.keys(), f"Gate {gate} not found in current gating strategy"
            gate = self.gates[gate]
        populations = self._apply(gate)
        if stats:
            data = self.get_population_df(population_name=gate.parent, transform=None)
            print(f"---- {gate.gate_name} outputs ----")
            print(f"Generated {len(populations)} populations:")
            for p in populations:
                print(f"{p.population_name}: n={len(p.index)}; {round(len(p.index) / data.shape[0] * 100, 3)}% of parent")
            print("----------------------------------")
        return self.plot_gate(gate=gate,
                              populations=populations,
                              create_plot_kwargs=create_plot_kwargs,
                              gate_plot_kwargs=plot_gate_kwargs)

    def apply(self,
              gate: Gate or str,
              plot_outcome: bool = True,
              create_plot_kwargs: dict or None = None,
              plot_gate_kwargs: dict or None = None):
        if isinstance(gate, str):
            assert gate in self.gates.keys(), f"Gate {gate} not found in current gating strategy"
            gate = self.gates[gate]
        assert gate.defined, "Gate children have not been labelled, call the 'label_children' " \
                             "method on the chosen Gate object"
        populations = self._apply(gate)
        for p in populations:
            self.populations[p.population_name] = p
            self.tree[p.population_name] = Node(name=p.population_name, parent=self.tree[gate.parent])
            if not self.crtl_gate_ad_hoc:
                self.control_gate(population=p,
                                  ctrl_id="all",
                                  plot_outcome=plot_outcome,
                                  verbose=self.verbose)
        if plot_outcome:
            self.plot_gate(gate=gate,
                           create_plot_kwargs=create_plot_kwargs,
                           gate_plot_kwargs=plot_gate_kwargs)
        self.gates[gate.gate_name] = gate

    def _apply_root_gates(self):
        root_gates = [g for g in self.gates.values() if g.parent == "root"]
        for gate in root_gates:
            self.vprint(f"-------------- {gate.gate_name} --------------")
            self.apply(gate=gate)
            self.vprint(f"----------------------------------------------")

    def _apply_action(self,
                      action: str,
                      error: bool = False):
        if all([p in self.populations.keys() for p in [self.actions.get(action).left,
                                                       self.actions.get(action).right]]):
            if self.actions.get(action).method == "merge":
                self.merge(left=self.actions.get(action).left,
                           right=self.actions.get(action).right,
                           new_population_name=self.actions.get(action).new_population_name)
            else:
                self.subtract(parent=self.actions.get(action).left,
                              targets=self.actions.get(action).right.split(","),
                              new_population_name=self.actions.get(action).new_population_name)
            return True
        if error:
            raise ValueError(f"Missing populations for the action {action}; "
                             f"[{self.actions.get(action).left}, {self.actions.get(action).right}]")
        return False

    def apply_all(self):
        assert len(self.gates) > 0, "No gates to apply"
        # First apply all gates that act on root
        self._apply_root_gates()
        # Then loop through all gates applying where parent exists
        downstream_gates = [g for g in self.gates.values() if g.parent != "root"]
        actions = list(self.actions.keys())
        i = 0
        iteration_limit = len(downstream_gates) * 100
        while len(downstream_gates) > 0:
            if i >= len(downstream_gates):
                i = 0
            gate = downstream_gates[i]
            if gate.parent in self.populations.keys():
                self.vprint(f"-------------- {gate.gate_name} --------------")
                parent_n = self.get_population_df(population_name=gate.parent).shape[0]
                if parent_n < 10:
                    warn(f"{gate.gate_name} parent population {gate.parent} contains less than 10 events, "
                         f"signifying an error upstream of this gate")
                self.apply(gate=downstream_gates[i])
                self.vprint(f"----------------------------------------------")
                downstream_gates = [x for x in downstream_gates if x.gate_name != gate.gate_name]
            applied_actions = list()
            for a in actions:
                action_applied = self._apply_action(a, error=False)
                if action_applied:
                    applied_actions.append(a)
            actions = [a for a in actions if a not in applied_actions]
            i += 1
            iteration_limit -= 1
            assert iteration_limit > 0, "Maximum number of iterations reached. This means that one or more parent " \
                                        "populations are not being identified."

    def _ctrl_training_data(self,
                            population: Population):
        training_data = self.get_population_df(population_name=population.parent,
                                               transform="logicle",
                                               transform_features="all")
        training_data["label"] = 0
        training_data.loc[population.index, "label"] = 1
        return training_data

    def _ctrl_optimal_n(self,
                        training_data: pd.DataFrame,
                        features: list):
        n = np.arange(int(training_data.shape[0] * 0.01),
                      int(training_data.shape[0] * 0.05),
                      int(training_data.shape[0] * 0.01) / 2, dtype=np.int)
        knn = KNeighborsClassifier()
        grid_cv = GridSearchCV(knn, {"n_neighbors": n}, scoring="balanced_accuracy", n_jobs=-1, cv=10)
        grid_cv.fit(training_data[features].values, training_data["label"].values)
        self.vprint(f"Continuing with n={n}; chosen with balanced accuracy of {round(grid_cv.best_score_, 3)}...")
        return grid_cv.best_params_.get("n_neighbors")

    def control_gate(self,
                     population: Population or str,
                     ctrl_id: str,
                     plot_outcome: bool = False,
                     verbose: bool = False):
        if isinstance(population, str):
            assert population in self.populations.keys(), f"No such population {population}"
            population = self.populations[population]
        if ctrl_id == "all":
            for ctrl in self.data.get("controls").keys():
                self.control_gate(population=population,
                                  ctrl_id=ctrl,
                                  plot_outcome=plot_outcome,
                                  verbose=verbose)
        self.vprint(f"---- Estimating {population.population_name} for {ctrl_id} ----")
        if self.populations.get(population.parent).ctrl_index.get(ctrl_id) is None:
            self.vprint(f"Missing data for parent {population.parent}....")
            self.control_gate(population=population.parent,
                              ctrl_id=ctrl_id,
                              plot_outcome=plot_outcome,
                              verbose=verbose)
        training_data = self._ctrl_training_data(population=population)
        ctrl_data = self.get_population_df(population_name=population.parent,
                                           transform="logicle",
                                           transform_features="all",
                                           ctrl_id=ctrl_id)
        x, y = population.geom.x, population.geom.y
        features = [x, y] if y is not None else [x]
        self.vprint("Calculating optimal n by cross-validation...")
        n = self._ctrl_optimal_n(training_data, features)
        self.vprint("Training on population data...")
        X_train, X_test, y_train, y_test = train_test_split(training_data[features].values,
                                                            training_data["label"].values,
                                                            test_size=0.2,
                                                            random_state=42)
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(X_train, y_train)
        train_acc = balanced_accuracy_score(y_pred=knn.predict(X_train), y_true=y_train)
        val_acc = balanced_accuracy_score(y_pred=knn.predict(X_test), y_true=y_test)
        self.vprint(f"...training balanced accuracy score: {train_acc}")
        self.vprint(f"...validation balanced accuracy score: {val_acc}")
        self.vprint(f"Predicting {population.population_name} in {ctrl_id} control...")
        ctrl_data["label"] = knn.predict(ctrl_data[features].values)
        population.ctrl_index = (ctrl_id, ctrl_data[ctrl_data["label"] == 1].index.values)
        self.populations[population.population_name] = population
        self.vprint("-------------- Complete --------------")

    def _merge_checks(self,
                      left: str,
                      right: str,
                      save_to_actions: bool):
        assert left in self.populations.keys(), f"{left} does not exist"
        assert right in self.populations.keys(), f"{right} does not exist"
        action_name = f"merge_{left}_{right}"
        if save_to_actions:
            assert action_name not in self.actions.keys(), "Merge action already exists in gating strategy"
        assert self.populations.get(left).parent == self.populations.get(right).parent, \
            "Populations must have the same parent in order to merge"

    def merge(self,
              left: str,
              right: str,
              new_population_name: str,
              save_to_actions: bool = True):
        self._merge_checks(left, right, save_to_actions)
        action_name = f"merge_{left}_{right}"
        left, right = self.populations.get(left), self.populations.get(right)
        new_population = merge_populations(left=left,
                                           right=right,
                                           new_population_name=new_population_name)
        self.populations[new_population.population_name] = new_population
        self.tree[new_population.population_name] = Node(name=new_population.population_name,
                                                         parent=self.tree[left.parent])
        if save_to_actions:
            self.actions[action_name] = Action(action_name=action_name,
                                               method="merge",
                                               left=left.population_name,
                                               right=right.population_name,
                                               new_population_name=new_population_name)

    def _subtract_checks(self,
                         parent: str,
                         targets: List[str],
                         save_to_actions: bool):
        for x in [parent] + targets:
            assert x in self.populations.keys(), "One or more given populations does not exist"
        parent = self.populations.get(parent)
        targets = [self.populations.get(t) for t in targets]
        assert all([t.parent == parent.population_name for t in targets]), \
            "Target populations must all derive directly from the given parent"
        assert len(set([x.geom.transform_x for x in [parent] + targets])) == 1, \
            "All populations must have the same transformation in the X-axis"
        assert len(set([x.geom.transform_y for x in [parent] + targets])) == 1, \
            "All populations must have the same transformation in the Y-axis"
        assert all([x.geom.y is not None for x in [parent] + targets]), \
            "Subtractions can only be performed on 2 dimensional gates"
        # Check that this is a unique action (if save_to_actions is True)
        action_name = f"subtract_{parent}_{','.join([t.population_name for t in targets])}"
        if save_to_actions:
            assert action_name not in self.actions.keys(), "Subtract action already exists in gating strategy"
        # Provide appropriate warnings
        if any([len(t.ctrl_index) > 0 for t in targets]):
            warn("Associated control indexes are not copied to new population. "
                 "Repeat control gating on new population")
        if any([len(t.clusters) > 0 for t in targets]):
            warn("Associated clusters are not copied to new population. "
                 "Repeat control gating on new population")

    def subtract(self,
                 parent: str,
                 targets: List[str],
                 new_population_name: str,
                 save_to_actions: bool = True):
        # Check the populations are valid
        targets = sorted(targets)
        parent = self.populations.get(parent)
        targets = [self.populations.get(t) for t in targets]
        self._subtract_checks(parent, targets, save_to_actions)
        # Estimate new index
        target_idx = np.unique(np.concatenate([t.index for t in targets], axis=0))
        new_population_idx = np.setdiff1d(parent.index, target_idx)
        # Create new polygon geom
        parent_data = self.get_population_df(parent.population_name)[[parent.geom.x, parent.geom.y]]
        parent_data = parent_data[parent_data.index.isin(new_population_idx)]
        x_values, y_values = parent_data[parent.geom.x].values, parent_data[parent.geom.y].values
        new_population_geom = Polygon(x=parent.geom.x,
                                      y=parent.geom.y,
                                      transform_x=parent.geom.transform_x,
                                      transform_y=parent.geom.transform_y,
                                      x_values=x_values,
                                      y_values=y_values)
        new_population = Population(population_name=new_population_name,
                                    n=len(parent.index) - len(new_population_idx),
                                    parent=parent.population_name,
                                    warnings=[t.warnings for t in targets] + ["SUBTRACTED POPULATION"],
                                    index=new_population_idx,
                                    geom=new_population_geom)
        self.populations[new_population_name] = new_population
        self.tree[new_population_name] = Node(name=new_population_name,
                                              parent=self.tree[parent.population_name])
        if save_to_actions:
            if save_to_actions:
                action_name = f"subtract_{parent}_{','.join([t.population_name for t in targets])}"
                self.actions[action_name] = Action(action_name=action_name,
                                                   method="subtract",
                                                   left=parent.population_name,
                                                   right=",".join([t.population_name for t in targets]),
                                                   new_population_name=new_population_name)

    def edit_gate(self,
                  population: str,
                  new_geom: Polygon or Threshold,
                  plot_outcome: bool = True,
                  create_plot_kwargs: dict or None = None,
                  plot_population_geom_kwargs: dict or None = None):
        create_plot_kwargs = create_plot_kwargs or {}
        plot_population_geom_kwargs = plot_population_geom_kwargs or {}
        assert population in self.populations.keys(), "Given population does not exist"
        dependecies = self.list_downstream_populations(population)
        ax = None
        if plot_outcome:
            plotting = CreatePlot(**create_plot_kwargs)
            ax = plotting.plot_population_geom(parent=self.get_population_df(self.populations[population].parent,
                                                                             transform=None),
                                               geom=new_geom,
                                               **plot_population_geom_kwargs)
        warn(f"The following populations are downstream from {population} and will be removed: {dependecies}")
        for p in dependecies:
            self.populations.pop(p)
            self.tree.pop(p)
        self.populations[population].geom = new_geom
        transforms = {new_geom.x: new_geom.transform_x}
        if new_geom.y:
            transforms[new_geom.y] = new_geom.transform_y
        parent = self.get_population_df(self.populations[population].parent,
                                        transform=transforms)
        if isinstance(new_geom, Threshold):
            self.populations[population].index = _edit_threshold_idx(parent, population, new_geom)
        else:
            self.populations[population].index = inside_polygon(df=parent, x=new_geom.x, y=new_geom.y, poly=new_geom.shape).index
        if not self.crtl_gate_ad_hoc:
            self.control_gate(self.populations[population], ctrl_id="all", plot_outcome=plot_outcome)
        else:
            warn(f"{population} in control files will need to be estimated again, if required")
        if len(self.populations[population].clusters) > 0:
            warn(f"{population} associated clusters will be removed")
            self.populations[population].clusters = []
        if plot_outcome:
            return ax

    def remove_population(self,
                          population: str):
        if population not in self.populations.keys():
            warn(f"{population} does not exist")
            return None
        dependencies = self.list_downstream_populations(population)
        if dependencies:
            warn(f"The following populations are downstream from {population} and will also be removed: {dependencies}")
        for p in dependencies:
            self.populations.pop(p)
            self.tree.pop(p)
        self.populations.pop(population)
        self.tree.pop(population)

    def remove_gate(self,
                    gate_name: str):
        assert gate_name in self.gates.keys(), f"{gate_name} does not exist"
        gate = self.gates.get(gate_name)
        if any([c in self.populations.keys() for c in gate.children]):
            dependences = [self.list_downstream_populations(c) for c in self.gates.get(gate_name).children]
            dependences = gate.children + [x in sl for sl in dependences for x in sl]
            warn(f"The following populations are a direct result of this gate and will be removed {dependences}")
            for p in dependences:
                self.populations.pop(p)
                self.tree.pop(p)
        self.gates.pop(gate_name)

    def print_population_tree(self,
                              image: bool = False,
                              image_name: str or None = None):
        root = self.tree['root']
        if image:
            if image_name is None:
                image_name = f'{self.id}_population_tree.png'
            DotExporter(root).to_picture(image_name)
        for pre, fill, node in RenderTree(root):
            print('%s%s' % (pre, node.name))


def load_population(sample_id: str,
                    experiment: Experiment,
                    population: str,
                    sample_n: int or None = None,
                    ctrl_id: str or None = None,
                    transform: str or None = None,
                    transform_features: list or str or dict = "all",
                    indexed: bool = False,
                    indexed_in_dataframe: bool = False):
    include_ctrls = ctrl_id is not None
    gating = Gating(experiment=experiment,
                    sample_id=sample_id,
                    include_controls=include_ctrls,
                    verbose=False)
    data = gating.get_population_df(population_name=population,
                                    transform=transform,
                                    transform_features=transform_features,
                                    ctrl_id=ctrl_id)
    if sample_n is not None:
        if data.shape[0] < sample_n:
            warn(f"{sample_id} has less than {sample_n} events (n={data.shape[0]}). Using all available data.")
        else:
            data = data.sample(n=sample_n)
    if indexed:
        if indexed_in_dataframe:
            data["sample_id"] = sample_id
            return data
        return sample_id, data
    return data


def check_population_tree(gating: Gating,
                          populations: list):
    """
    Check that a given list of population names follows the hierarchy described in the given Gating object

    Parameters
    ----------
    gating
    populations

    Returns
    -------

    """
    assert all([p in gating.populations.keys() for p in populations]), "One or more given populations does not exist " \
                                                                       "in the Gating object"
    root = populations[0]
    populations = populations[1:]
    assert all([x in gating.list_downstream_populations(root) for x in populations]), \
        "Root population does not contain all the subsequent populations provided in the ordered list 'populations'"
    for i, pop in enumerate(populations):
        if i == len(populations) - 1:
            continue
        assert not any([x in gating.list_dependencies(population=pop) for x in populations[i + 1:]]), \
            "Population list is not ordered; one or more populations follows a population to which it is dependent"
