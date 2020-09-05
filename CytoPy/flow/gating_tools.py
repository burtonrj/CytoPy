from ..data.experiments import Experiment
from ..data.populations import PopulationGeometry, Population, merge_populations
from ..data.gates import Gate, PreProcess, PostProcess
from ..data.gating_strategy import GatingStrategy, Action
from .transforms import apply_transform
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
    valid_clusters = [x[0] for x in inspect.getmembers(cluster, inspect.isclass)
                      if 'sklearn.cluster' in x[1].__module__]
    valid_mixtures = [x[0] for x in inspect.getmembers(mixture, inspect.isclass)
                      if 'sklearn.mixture' in x[1].__module__]
    valid = valid_clusters + valid_mixtures + ["HDBSCAN"]
    err = f"""Invalid class name. Must be one of the following from Scikit-Learn's cluster module: {valid_clusters};
 or from Scikit-Learn's mixture module: {valid_mixtures}; or 'HDBSCAN'"""
    assert klass in valid, err
    return klass


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
    sample_n: int, optional
        number of events to sample from FCS file(s) (optional)
    include_controls: bool, (default=True)
        if True and FMOs are included for specified samples, the FMO data will also be loaded into the Gating object
    """

    def __init__(self,
                 experiment: Experiment,
                 sample_id: str,
                 gating_strategy: str or None = None,
                 include_controls=True,
                 verbose: bool = True,
                 gate_ctrls_adhoc: bool = True):
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
        if not self.filegroup.populations:
            # No population currently exist for this FileGroup. Init with root population
            self.populations = {"root": Population(population_name="root",
                                                   index=self.data.get("primary").index.values,
                                                   parent="root",
                                                   n=len(self.data.get("primary").index.values))}
            if "controls" in self.data.keys():
                for ctrl_id, ctrl_data in self.data.get("controls").items():
                    self.populations["root"].ctrl_index = (ctrl_id, ctrl_data.index.values)
            return {"root": Node(name="root", parent=None)}
        self.populations["root"] = self.filegroup.get_population("root")
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
        if new_population.parent not in tree.keys():
            return None
        tree[new_population.population_name] = Node(name=new_population.population_name,
                                                    parent=new_population.parent)
        return tree

    def save_sample(self,
                    overwrite: bool = False):
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
            GatingStrategy.objects(template_name=self.template.template_name).get().delete()
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
                          transform_features: list or str = 'all',
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
        root = self.populations['root']
        node = self.populations[population]
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
        features_to_check = [x]
        if y is not None:
            features_to_check.append(y)
        if any(c not in self.data.get("primary").columns for c in features_to_check):
            if not preprocessing_kwargs.get("dim_reduction"):
                err = f"x or y are invalid values are invalid; valid column names as: {self.data.get('primary').columns}"
                raise ValueError(err)
            else:
                assert x == "embedding1", "If using dim_reduction, x should have a value 'embedding1'"
                assert y == "embedding2", "If using dim_reduction, y should have a value 'embedding2'"

        if method == "ManualGate":
            assert binary, "ManualGate is for use with binary gates only"
        elif shape == "threshold":
            if method != "DensityGate":
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
            method = valid_sklearn(method)
            if "dbscan" in method.lower():
                if preprocessing_kwargs.get("downsample_method") is None:
                    warn("DBSCAN and HDBSCAN do not scale well and it is recommended that downsampling is performed")
        gate = Gate(gate_name=gate_name,
                    parent=parent,
                    shape=shape,
                    x=x,
                    y=y,
                    binary=binary,
                    method=method,
                    method_kwargs=[(k, v) for k, v in method_kwargs.items()],
                    preprocessing=PreProcess(**preprocessing_kwargs),
                    postprocessing=PostProcess(**postprocessing_kwargs))
        return gate

    def plot_gate(self,
                  gate: Gate,
                  create_plot_kwargs: dict or None = None,
                  gate_plot_kwargs: dict or None = None,
                  populations: list or None = None):
        if create_plot_kwargs is None:
            create_plot_kwargs = {}
        if gate_plot_kwargs is None:
            gate_plot_kwargs = {}
        data = self.get_population_df(population_name=gate.parent, transform=None)
        if populations is None:
            children = self.list_child_populations(gate.parent)
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

    def preview(self,
                gate: Gate,
                verbose: bool = True,
                stats: bool = True,
                create_plot_kwargs: dict or None = None,
                plot_gate_kwargs: dict or None = None):
        data = self.get_population_df(population_name=gate.parent, transform=None)
        ctrl = None
        if gate.ctrl_id:
            ctrl = self.get_population_df(population_name=gate.parent, transform=None, ctrl_id=gate.ctrl_id)
        populations = gate.apply(data=data,
                                 ctrl=ctrl,
                                 verbose=verbose)
        if stats:
            print(f"---- {gate.gate_name} outputs ----")
            print(f"Generated {len(populations)} populations:")
            for p in populations:
                print(
                    f"{p.population_name}: n={len(p.index)}; {round(len(p.index) / data.shape[0] * 100, 3)}% of parent")
            print("----------------------------------")
        return self.plot_gate(gate=gate,
                              populations=populations,
                              create_plot_kwargs=create_plot_kwargs,
                              gate_plot_kwargs=plot_gate_kwargs)

    def apply(self,
              gate: Gate or None = None,
              gate_name: str or None = None,
              verbose: bool = True,
              plot_outcome: bool = True,
              create_plot_kwargs: dict or None = None,
              plot_gate_kwargs: dict or None = None):
        if gate is None and gate_name is None:
            raise ValueError("Must provide Gate object or name of an existing gate in the loaded template")
        if gate is None:
            assert gate_name in self.gates.keys(), f"Invalid gate, must be one of: {self.gates.keys()}"
            gate = self.gates.get(gate_name)
        assert gate.defined, "Gate children have not been labelled, call the 'label_children' " \
                             "method on the chosen Gate object"
        data = self.get_population_df(population_name=gate.parent, transform=None)
        ctrl = None
        if gate.ctrl_id:
            ctrl = self.get_population_df(population_name=gate.parent, transform=None, ctrl_id=gate.ctrl_id)
        populations = gate.apply(data=data,
                                 ctrl=ctrl,
                                 verbose=verbose)
        for p in populations:
            self.populations[p.population_name] = p
            self.tree[p.population_name] = Node(name=p.population_name, parent=self.tree[gate.parent])
            if not self.crtl_gate_ad_hoc:
                self.control_gate(population=p,
                                  ctrl_id="all",
                                  plot_outcome=plot_outcome,
                                  verbose=verbose)
        if plot_outcome:
            self.plot_gate(gate=gate,
                           create_plot_kwargs=create_plot_kwargs,
                           gate_plot_kwargs=plot_gate_kwargs)
        self.gates[gate.gate_name] = gate

    def apply_all(self,
                  verbose: bool = True,
                  return_plots: bool = True,
                  create_plot_kwargs: dict or None = None,
                  plot_gate_kwargs: dict or None = None):
        assert len(self.gates) > 0, "No gates to apply"
        plots = list()
        feedback = vprint(verbose)
        # First apply all gates that act on root
        root_gates = [g for g in self.gates.values() if g.parent == "root"]
        for gate in root_gates:
            feedback(f"-------------- {gate.gate_name} --------------")
            plots.append(self.apply(gate=gate,
                                    verbose=verbose,
                                    plot_outcome=return_plots,
                                    create_plot_kwargs=create_plot_kwargs,
                                    plot_gate_kwargs=plot_gate_kwargs))
            feedback(f"----------------------------------------------")
        # Then loop through all gates applying where parent exists
        downstream_gates = [g for g in self.gates.values() if g.parent != "root"]
        actions = list(self.actions.keys())
        i = 0
        while len(downstream_gates) > 0:
            if i >= len(downstream_gates):
                i = 0
            if return_plots:
                gate = downstream_gates[i]
                if gate.parent in self.populations.keys():
                    feedback(f"-------------- {gate.gate_name} --------------")
                    plots.append(self.apply(gate=downstream_gates[i],
                                            verbose=verbose,
                                            plot_outcome=return_plots,
                                            create_plot_kwargs=create_plot_kwargs,
                                            plot_gate_kwargs=plot_gate_kwargs))
                    feedback(f"----------------------------------------------")
            applied_actions = list()
            for a in actions:
                if all([p in self.populations.keys() for p in [self.actions.get(a).left,
                                                               self.actions.get(a).right]]):
                    applied_actions.append(a)
                    if self.actions.get("a").method == "merge":
                        self.merge(left=self.actions.get(a).left,
                                   right=self.actions.get(a).right,
                                   new_population_name=self.actions.get(a).new_population_name)
                    else:
                        self.subtract(parent=self.actions.get(a).left,
                                      targets=self.actions.get(a).right.split(","),
                                      new_population_name=self.actions.get(a).new_population_name)
            actions = [a for a in actions if a not in applied_actions]
            i += 1
        if return_plots:
            return plots

    def control_gate(self,
                     population: Population,
                     ctrl_id: str,
                     plot_outcome: bool = False,
                     verbose: bool = False):
        if ctrl_id == "all":
            for ctrl in self.data.get("controls").keys():
                self.control_gate(population=population,
                                  ctrl_id=ctrl,
                                  plot_outcome=plot_outcome,
                                  verbose=verbose)
        feedback = vprint(verbose)
        feedback(f"---- Estimating {population.population_name} for {ctrl_id} ----")
        if self.populations.get(population.parent).ctrl_index.get(ctrl_id) is None:
            feedback(f"Missing data for parent {population.parent}....")
            self.control_gate(population=population.parent,
                              ctrl_id=ctrl_id,
                              plot_outcome=plot_outcome,
                              verbose=verbose)
        training_data = self.get_population_df(population_name=population.parent,
                                               transform="logicle",
                                               transform_features="all")
        training_data["label"] = 0
        training_data.loc[population.index, "label"] = 1
        ctrl_data = self.get_population_df(population_name=population.parent,
                                           transform="logicle",
                                           transform_features="all",
                                           ctrl_id=ctrl_id)
        x, y = population.geom.x, population.geom.y
        features = [x, y] if y is not None else [x]
        feedback("Calculating optimal n by cross-validation...")
        n = np.arange(int(training_data.shape[0] * 0.01),
                      int(training_data.shape[0] * 0.05),
                      int(training_data.shape[0] * 0.01) / 2, dtype=np.int)
        knn = KNeighborsClassifier()
        grid_cv = GridSearchCV(knn, {"n": n}, scoring="balanced_accuracy", n_jobs=-1, cv=10)
        grid_cv.fit(training_data[features].values, training_data["label"].values)
        n = grid_cv.best_params_.get("n")
        feedback(f"Continuing with n={n}; chosen with balanced accuracy of {round(grid_cv.best_score_, 3)}...")
        feedback("Training on population data...")
        X_train, X_test, y_train, y_test = train_test_split(training_data[features].values,
                                                            training_data["label"].values,
                                                            test_size=0.2,
                                                            random_state=42)
        knn = KNeighborsClassifier(n=n)
        knn.fit(X_train, y_train)
        train_acc = balanced_accuracy_score(y_pred=knn.predict(X_train), y_true=y_train)
        val_acc = balanced_accuracy_score(y_pred=knn.predict(X_test), y_true=y_test)
        feedback(f"...training balanced accuracy score: {train_acc}")
        feedback(f"...validation balanced accuracy score: {val_acc}")
        feedback(f"Predicting {population.population_name} in {ctrl_id} control...")
        ctrl_data["label"] = knn.predict(ctrl_data[features].values)
        population.ctrl_index = (ctrl_id, ctrl_data[ctrl_data["label"] == 1].index.values)
        self.populations[population.population_name] = population
        feedback("-------------- Complete --------------")

    def merge(self,
              left: str,
              right: str,
              new_population_name: str,
              save_to_actions: bool = True):
        assert left in self.populations.keys(), f"{left} does not exist"
        assert right in self.populations.keys(), f"{right} does not exist"
        action_name = f"merge_{left}_{right}"
        if save_to_actions:
            assert action_name not in self.actions.keys(), "Merge action already exists in gating strategy"
        left, right = self.populations.get(left), self.populations.get(right)
        assert left.parent == right.parent, "Populations must have the same parent in order to merge"
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

    def subtract(self,
                 parent: str,
                 targets: List[str],
                 new_population_name: str,
                 save_to_actions: bool = True):
        # Check the populations are valid
        targets = sorted(targets)
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
        action_name = f"subtract_{parent}_{','.join(targets)}"
        if save_to_actions:
            assert action_name not in self.actions.keys(), "Subtract action already exists in gating strategy"
        # Provide appropriate warnings
        if any([len(t.ctrl_index) > 0 for t in targets]):
            warn("Associated control indexes are not copied to new population. "
                 "Repeat control gating on new population")
        # TODO lookup all clusters applied to this population and delete
        warn("Associated clusters are not copied to new population. "
             "Repeat control gating on new population")
        # Estimate new index
        target_idx = np.unique(np.concatenate([t.index for t in targets], axis=0))
        new_population_idx = np.setdiff1d(parent.index, target_idx)
        # Create new polygon geom
        parent_data = self.get_population_df(parent.population_name)[[parent.geom.x, parent.geom.y]]
        parent_data = parent_data[parent_data.index.isin(new_population_idx)]
        x_values, y_values = parent_data[parent.geom.x].values, parent_data[parent.geom.y].values
        new_population_geom = PopulationGeometry(x=parent.geom.x,
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
                self.actions[action_name] = Action(action_name=action_name,
                                                   method="subtract",
                                                   left=parent.population_name,
                                                   right=",".join([t.population_name for t in targets]),
                                                   new_population_name=new_population_name)

    def edit_gate(self,
                  population: str,
                  new_geom: PopulationGeometry or None = None,
                  plot_outcome: bool = True,
                  create_plot_kwargs: dict or None = None,
                  plot_population_geom_kwargs: dict or None = None):
        if create_plot_kwargs is None:
            create_plot_kwargs = {}
        if plot_population_geom_kwargs is None:
            plot_population_geom_kwargs = {}
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
        if not self.crtl_gate_ad_hoc:
            self.control_gate(self.populations[population], ctrl_id="all", plot_outcome=plot_outcome)
        else:
            warn(f"{population} in control files will need to be estimated again, if required")
        # TODO remove clusters
        if plot_outcome:
            return ax

    def remove_population(self,
                          population: str):
        if population not in self.populations.keys():
            warn(f"{population} does not exist")
            return None
        dependecies = self.list_downstream_populations(population)
        if dependecies:
            warn(f"The following populations are downstream from {population} and will also be removed: {dependecies}")
        for p in dependecies:
            self.populations.pop(p.population_name)
            self.tree.pop(p.population_name)

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
                    transform_features: list or str = "all",
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
