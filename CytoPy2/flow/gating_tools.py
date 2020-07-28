from ..data.experiments import Experiment
from ..data.fcs import Population, PopulationGeometry
from ..data.gates import Gate, PreProcess, PostProcess
from .dim_reduction import dimensionality_reduction
from .transforms import apply_transform, scaler
from ..feedback import progress_bar, vprint
from .sampling import density_dependent_downsampling, faithful_downsampling
from .plotting import CreatePlot
from anytree import Node, findall
from warnings import warn
import pandas as pd


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
    sample: int, optional
        number of events to sample from FCS file(s) (optional)
    include_controls: bool, (default=True)
        if True and FMOs are included for specified samples, the FMO data will also be loaded into the Gating object
    """

    def __init__(self,
                 experiment: Experiment,
                 sample_id: str,
                 sample: int or None = None,
                 include_controls=True,
                 verbose: bool = True):
        data = experiment.get_data(sample_id=sample_id, sample_size=sample, include_controls=include_controls)
        assert data is not None, f'Error: failed to fetch data for {sample_id}. Aborting.'
        self.data = [x for x in data
                     if x['typ'] == 'complete'][0]['data']
        if include_controls:
            self.ctrl = [x for x in data
                         if x['typ'] == 'control']
            self.ctrl = {x['id'].replace(f'{sample_id}_', ''): x['data']
                         for x in self.ctrl}
        else:
            self.ctrl = {}
        del data
        self.id = sample_id
        self.mongo_id = experiment.get_sample_mid(sample_id)
        self.experiment = experiment
        self.filegroup = experiment.get_sample(sample_id)
        self.gates = dict()
        self.populations = dict()
        self.tree = self._construct_tree()
        self.verbose = verbose
        self.vprint = vprint(verbose)
        self.preview_cache = None

    def _construct_tree(self):
        if not self.filegroup.populations:
            # No population currently exist for this FileGroup. Init with root population
            self.populations = {"root": Population(population_name="root",
                                                   index=self.data.index.values,
                                                   parent="root")}
            if self.ctrl:
                for ctrl_id, ctrl_data in self.ctrl.items():
                    self.populations["root"]["control_idx"].append(ControlIndex(control_id=ctrl_id,
                                                                                index=ctrl_data.index.values))
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

    def save(self):
        # SAVE GATING FOR SAMPLE
        pass

    def save_template(self):
        pass

    def load_template(self):
        pass

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
            data = self.data.loc[idx]
        else:
            idx = self.populations[population_name].get_ctrl(population_name)
            assert idx is not None, f'No cached index for {ctrl_id} associated to population {population_name}, ' \
                                    f'have you called "control_gating" previously?'
            data = self.ctrl[ctrl_id].loc[idx]
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
        dependencies = self.find_dependencies(population_name)
        for pop in dependencies:
            idx = self.populations[pop].index
            data.loc[idx, 'label'] = pop
        return data

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
        dependencies = [x.name for x in findall(root, filter_=lambda n: node in n.path)]
        return [p for p in dependencies if p != population]

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
        assert gate_name not in self.gates.keys(), f"{gate_name} already exists!"
        err = """Gate should have one of the following shapes: ["threshold", "polygon", "ellipse"]"""
        assert shape in ["threshold", "polygon", "ellipse"], err
        assert parent in self.populations.keys(), "Invalid parent (does not exist)"
        if any(c not in self.data.columns for c in [x, y]):
            if not preprocessing_kwargs.get("dim_reduction"):
                raise ValueError("x or y are invalid given that dim_reduction is None or undefined")
            else:
                assert x == "embedding1", "If using dim_reduction, x should have a value 'embedding1'"
                assert y == "embedding2", "If using dim_reduction, x should have a value 'embedding2'"
        if shape == "threshold":
            if "method" != "DensityGate":
                warn("Shape set to 'threshold', defaulting to DensityGate")
            method = "DensityGate"
        if shape == "ellipse":
            err = "For an elliptical gate, expect method 'GaussianMixture', 'BayesianGaussianMixture', " \
                  "or 'MiniBatchKMeans'"
            assert method in ["GaussianMixture", "BayesianGaussianMixture", 'MiniBatchKMeans'], err
            if method is None:
                warn("Method not given, defaulting to BayesianGaussianMixture")
                method = "BayesianGaussianMixture"
                method_kwargs = {"n_com"}
        if shape == "polygon":
            accepted_methods = ["Affinity",
                                "Hierarchical",
                                "Birch",
                                "Dbscan",
                                "Hdbscan",
                                "MeanShift",
                                "Spectral"]
            err = f"For a polygon gate, accepted methods are: {accepted_methods}"
            assert method in accepted_methods, err
            assert method, "For a polygon gate, the user must specify the method"
        gate = Gate(gate_name=gate_name,
                    parent=parent,
                    geom_type=shape,
                    x=x,
                    y=y,
                    binary=binary,
                    method=method,
                    method_kwargs=[(k, v) for k, v in method_kwargs.items()],
                    preprocessing_kwargs=PreProcess(**preprocessing_kwargs),
                    postprocessing_kwargs=PostProcess(**postprocessing_kwargs))
        gate.initialise_model()
        return gate

    def preview(self,
                gate: Gate,
                stats: bool = True):
        data = gate.execute_preprocessing(data=self.get_population_df(population_name=gate.parent,
                                                                      transform=None))
        # fit model
        populations = gate.model.fit(data)
        # Cache results
        self.preview_cache = {"data": data,
                              "populations": populations}
        # Plot with CreatePlot, complete with legend
        plotting = CreatePlot()
        # Print stats
        return gate

    def apply(self,
              gate: Gate or None = None,
              gate_name: str or None = None,
              labels: dict or None = None,
              plot_outcome: bool = True,
              preprocess_kwargs: dict or None = None,
              method_kwargs: dict or None = None,
              postprocess_kwargs: dict or None = None):
        # If Gate, check cache. If cached, label populations using labels and save
        # If Gate but not cache, fit and then label using labels, then save
        # If gate name, load and init Gate object. Fit model and estimate population labels, save.
        # Plot if specified
        return gate

    def control_gate(self):
        pass

    def merge(self):
        pass

    def subtract(self):
        pass

    def apply_all(self):
        # Apply all gates
        pass

    def edit_gate(self,
                  new_geom: PopulationGeometry or None = None,
                  new_x: float or None = None,
                  new_y: float or None = None):
        pass

    def remove_population(self):
        pass

    def remove_gate(self):
        pass

    def print_population_tree(self):
        pass

    def check_downstream_overlaps(self):
        pass
