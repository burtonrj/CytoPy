#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
In a traditional analysis, an immunologist would apply a 'gating strategy';
a series of 'gates' that separate single cell data into the populations of
interest. cytopy provides autonomous gates (see cytopy.data.gate) to
emulate this process and these gates can be packaged together for bulk
analysis using the GatingStrategy class, housed within this module.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from cytopy.flow.plotting.flow_plot import FlowPlot
from ..feedback import progress_bar, vprint
from .gate import Gate, ThresholdGate, PolygonGate, EllipseGate, ThresholdGeom, \
    PolygonGeom, update_polygon, update_threshold, BooleanGate
from ..flow.transform import apply_transform
from ..flow.gate_search import hyperparameter_gate
from ..flow.fda_norm import LandmarkReg
from .experiment import Experiment
from .fcs import FileGroup
from .errors import *
from datetime import datetime
from warnings import warn
from matplotlib import gridspec
from typing import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mongoengine
import logging
import math
import os

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"
logger = logging.getLogger("GatingStrategy")


def gate_stats(gate: Gate,
               populations: list,
               parent_data: pd.DataFrame):
    """
    Print the statistics of populations generated from a Gate

    Parameters
    ----------
    gate: Gate
    populations: list
        List of populations generated from fit_predict method of a Gate
    parent_data: Pandas.DataFrame
        Parent data that the gate is applied to

    Returns
    -------
    None
    """
    print(f"----- {gate.gate_name} -----")
    parent_n = parent_data.shape[0]
    print(f"Parent ({gate.parent}) n: {parent_n}")
    for p in populations:
        print(f"...child {p.population_name} n: {p.n}; {p.n / parent_n * 100}% of parent")
    print("------------------------")


class GatingStrategy(mongoengine.Document):
    """
    A GatingTemplate is synonymous to what an immunologist would classically consider
    a "gating template"; it is a collection of 'gates' (Gate objects, in the case of cytopy)
    that can be applied to multiple fcs files or an entire experiment in bulk. A user defines
    a GatingTemplate using a single example from an experiment, uses the object to preview gates
    and label child populations, and when satisfied with the performance save the GatingStrategy
    to the database to be applied to the remaining samples in the Experiment.

    Attributes
    -----------
    name: str, required
        unique identifier for template
    gates: EmbeddedDocumentList
        list of Gate documents
    creation_date: DateTime
        date of creation
    last_edit: DateTime
        date of last edit
    flags: str, optional
        warnings associated to this gating template
    notes: str, optional
        free text comments
    hyperparameter_search: dict
        Hyperparameter search definitions; populated using the add_hyperparameter_grid method
    normalisation: dict
        Normalisation definitions; populated using the add_normalisation method
    """
    name = mongoengine.StringField(required=True, unique=True)
    gates = mongoengine.ListField(mongoengine.ReferenceField(Gate, reverse_delete_rule=mongoengine.PULL))
    hyperparameter_search = mongoengine.DictField()
    normalisation = mongoengine.DictField()
    creation_date = mongoengine.DateTimeField(default=datetime.now)
    last_edit = mongoengine.DateTimeField(default=datetime.now)
    flags = mongoengine.StringField(required=False)
    notes = mongoengine.StringField(required=False)
    meta = {
        'db_alias': 'core',
        'collection': 'gating_strategy'
    }

    def __init__(self, *args, **values):
        self.verbose = values.pop("verbose", True)
        super().__init__(*args, **values)
        self.filegroup = None

    def load_data(self,
                  experiment: Experiment,
                  sample_id: str):
        """
        Load a FileGroup into the GatingStrategy ready for gating.

        Parameters
        ----------
        experiment: Experiment
        sample_id: str

        Returns
        -------
        None
        """
        self.filegroup = experiment.get_sample(sample_id=sample_id)

    def list_gates(self) -> list:
        """
        List name of existing Gates

        Returns
        -------
        list
        """
        return [g.gate_name for g in self.gates]

    def list_populations(self) -> list:
        """
        Wrapper to FileGroup list_populations. Lists populations
        in associated FileGroup.

        Returns
        -------
        list

        Raises
        ------
        AssertionError
            No FileGroup loaded
        """
        assert self.filegroup is not None, "No FileGroup associated"
        return list(self.filegroup.list_populations())

    def _gate_exists(self,
                     gate: str):
        """
        Raises AssertionError if given gate does not exist

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            Gate does not exist
        """
        assert gate in self.list_gates(), f"Gate {gate} does not exist"

    def get_gate(self,
                 gate: str) -> Gate:
        """
        Given the name of a gate, return the Gate object

        Parameters
        ----------
        gate: str

        Returns
        -------
        Gate
        """
        self._gate_exists(gate=gate)
        return [g for g in self.gates if g.gate_name == gate][0]

    def preview_gate(self,
                     gate: str or ThresholdGate or PolygonGate or EllipseGate,
                     create_plot_kwargs: Optional[Dict] = None,
                     plot_gate_kwargs: Optional[Dict] = None):
        """
        Preview the results of some given Gate

        Parameters
        ----------
        gate: str or ThresholdGate or PolygonGate or EllipseGate
            Name of an existing Gate or a Gate object
        create_plot_kwargs: dict (optional)
            Additional arguments passed to CreatePlot
        plot_gate_kwargs: dict (optional)
            Additional arguments passed to plot_gate call of CreatePlot

        Returns
        -------
        Matplotlib.Axes
        """
        create_plot_kwargs = create_plot_kwargs or {}
        plot_gate_kwargs = plot_gate_kwargs or {}
        if isinstance(gate, str):
            gate = self.get_gate(gate=gate)
        if isinstance(gate, BooleanGate):
            data, plot_data = self._load_gate_dataframes_boolean(gate=gate)
            ctrl_parent_data = None
        else:
            data, ctrl_parent_data = self._load_gate_dataframes(gate=gate, fda_norm=False)
            plot_data = data
        gate.fit(data=data, ctrl_data=ctrl_parent_data)
        create_plot_kwargs["transform_x"] = create_plot_kwargs.get("transform_x", None) or gate.transform_x
        create_plot_kwargs["transform_y"] = create_plot_kwargs.get("transform_y", None) or gate.transform_y
        create_plot_kwargs["transform_x_kwargs"] = create_plot_kwargs.get("transform_x_kwargs",
                                                                          None) or gate.transform_x_kwargs
        create_plot_kwargs["transform_y_kwargs"] = create_plot_kwargs.get("transform_y_kwargs",
                                                                          None) or gate.transform_y_kwargs
        plot = FlowPlot(**create_plot_kwargs)
        return plot.plot_gate_children(gate=gate,
                                       parent=plot_data,
                                       **plot_gate_kwargs)

    def add_hyperparameter_grid(self,
                                gate_name: str,
                                params: dict,
                                cost: Optional[str] = None):
        """
        Add a hyperparameter grid to search when applying the given gate to new data.
        This hyperparameter grid should correspond to valid hyperparameters for the
        corresponding gate. Invalid parameters will be ignored. Choice of the cost
        parameter to be minimised is dependent on the type of gate:
        * ThresholdGate:
            - "manhattan" (default): optimal parameters are those that result in the population whom's signature
              is of minimal distance to the original data used to define the gate. The manhattan distance is used
              as the distance metric.
            - "euclidean": optimal parameters are those that result in the population whom's signature
              is of minimal distance to the original data used to define the gate. The euclidean distance is used
              as the distance metric.
            - "threshold_dist": optimal parameters are those that result in the threshold
               whom's distance to the original threshold defined are smallest
        * PolygonGate & EllipseGate:
            - "hausdorff" (optional): parameters chosen that minimise the hausdorff distance
              between the polygon generated from new data and the original polgon gate created
              when the gate was defined
            - "manhattan" (default): optimal parameters are those that result in the population whom's signature
              is of minimal distance to the original data used to define the gate. The manhattan distance is used
              as the distance metric.
            - "euclidean": optimal parameters are those that result in the population whom's signature
              is of minimal distance to the original data used to define the gate. The euclidean distance is used
              as the distance metric.

        Parameters
        ----------
        gate_name: str
            Gate to define hyperparameter grid for
        params: dict
            Grid of hyperparameters to be searched
        cost: str
            What to be minimised to choose optimal hyperparameters

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            Invalid metric or hyperparameters
        """
        assert gate_name in self.list_gates(), f"{gate_name} is not a valid gate"
        if isinstance(self.get_gate(gate_name), ThresholdGate):
            cost = cost or "manhattan"
            valid_metrics = ["manhattan", "threshold_dist", "euclidean"]
            err = f"For threshold gate 'cost' should either be one of {valid_metrics}"
            assert cost in valid_metrics, err
        if isinstance(self.get_gate(gate_name), PolygonGate) or isinstance(self.get_gate(gate_name), EllipseGate):
            cost = cost or "hausdorff"
            valid_metrics = ["hausdorff", "manhattan", "euclidean"]
            err = f"For threshold gate 'cost' should either be one of {valid_metrics}"
            assert cost in valid_metrics, err
        err = "'params' must be a dictionary with each key corresponding to a valid " \
              "hyperparameter and each value a list of parameter values"
        assert isinstance(params, dict), err
        assert all([isinstance(x, list) for x in params.values()]), err
        self.hyperparameter_search[gate_name] = {"grid": params,
                                                 "cost": cost}

    def add_normalisation(self,
                          gate_name: str,
                          reference: Union[FileGroup, None] = None,
                          **kwargs):
        """
        Add landmark registration for normalisation to a Gate. In short, if normalisation is added
        to a gate (specified by 'gate_name') data that this gate is applied too, will be 'aligned'
        to some reference data (specified by 'reference', which should be a FileGroup object, but
        if left as None, will be the FileGroup currently associated with the GatingStrategy).
        Alignment is performed using a peak finding algorithm, K means clustering, and then
        landmark registration; see cytopy.flow.fda_norm for details.

        Parameters
        ----------
        gate_name: str
            Name of the gate to assign normalisation to
        reference: FileGroup, optional
            FileGroup that will be used as reference data for future data to be aligned. If
            value is None (default) then the currently associated FileGroup to this GatingStrategy
            is used as the future reference
        kwargs:
            Additional keyword arguments that will be passed to cytopy.flow.fda_norm.LandmarkReg

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If invalid gate name provided
        """
        reference = reference or self.filegroup
        assert gate_name in self.list_gates(), "Invalid gate name"
        self.normalisation[gate_name] = {"reference": str(reference.id),
                                         "kwargs": kwargs}

    def normalise_data(self,
                       gate_name: str):
        """
        Given the name of an existing Gate, perform normalisation of the parent
        population using the details specified in the normalisation attribute. If
        no normalisation has been specified for the given gate, the parent population
        will be returned without normalisation performed.

        Parameters
        ----------
        gate_name: str

        Returns
        -------
        Pandas.DataFrame
        """
        if gate_name not in self.normalisation.keys():
            return self._load_gate_dataframes(gate=self.get_gate(gate_name), fda_norm=False)[0]
        gate = self.get_gate(gate_name)
        if self.normalisation.get(gate_name).get("reference") == str(self.filegroup.id):
            return self._load_gate_dataframes(gate=self.get_gate(gate_name), fda_norm=False)[0]
        ref = FileGroup.objects(id=self.normalisation.get(gate_name).get("reference")).get()
        kwargs = self.normalisation.get(gate_name).get("kwargs")
        data = self.filegroup.load_population_df(population=gate.parent,
                                                 transform=None)
        for d, t, tkwargs in zip([gate.x, gate.y],
                                 [gate.transform_x, gate.transform_y],
                                 [gate.transform_x_kwargs, gate.transform_y_kwargs]):
            if d is None:
                continue
            ref_df = ref.load_population_df(population=gate.parent,
                                            transform=t,
                                            transform_kwargs=tkwargs)
            target_df = self.filegroup.load_population_df(population=gate.parent,
                                                          transform=None)
            target_df, transformer = apply_transform(data=target_df,
                                                     method=t,
                                                     return_transformer=True,
                                                     features=[d],
                                                     **tkwargs)
            try:
                lr = LandmarkReg(target=target_df,
                                 ref=ref_df,
                                 var=d,
                                 **kwargs)
                target_df[d] = lr().shift_data(target_df[d].values)
            except ValueError as e:
                warn(f"Failed to normalise data in {d} dimension, continuing without normalisation; "
                     f"{str(e)}")
            if transformer is not None:
                target_df = transformer.inverse_scale(data=target_df, features=[d])
            data[d] = target_df[d]
        return data

    def _load_gate_dataframes(self,
                              gate: Gate,
                              fda_norm: bool = False,
                              verbose: bool = True,
                              ctrl: bool = True):
        """
        Load the parent population dataframe ready for a Gate to be applied.

        Parameters
        ----------
        gate: Gate
            Gate that will be applied
        fda_norm: bool (default=False)
            Perform normalisation if defined for Gate
        verbose: bool (default=True)
            Provide standard feedback
        ctrl: bool (default=True)
            Load control population data if Gate is defined for control gating

        Returns
        -------
        Pandas.DataFrame, Pandas.DataFrame or None
            Parent population, control population (if Gate is a control gate, otherwise None)
        """
        parent = self.filegroup.load_population_df(population=gate.parent,
                                                   transform=None,
                                                   label_downstream_affiliations=False)
        if fda_norm:
            return self.normalise_data(gate_name=gate.gate_name), None
        if gate.ctrl_x is not None and ctrl:
            ctrls = {}
            ctrl_classifier_params = gate.ctrl_classifier_params or {}
            kwargs = gate.ctrl_prediction_kwargs or {}
            if not verbose:
                kwargs["verbose"] = False
            if gate.ctrl_x is not None:
                x = self.filegroup.load_ctrl_population_df(ctrl=gate.ctrl_x,
                                                           population=gate.parent,
                                                           classifier=gate.ctrl_classifier,
                                                           classifier_params=ctrl_classifier_params,
                                                           verbose=verbose,
                                                           **kwargs)
                ctrls[gate.ctrl_x] = x[gate.ctrl_x].values
            if gate.ctrl_y is not None:
                y = self.filegroup.load_ctrl_population_df(ctrl=gate.ctrl_y,
                                                           population=gate.parent,
                                                           classifier=gate.ctrl_classifier,
                                                           classifier_params=ctrl_classifier_params,
                                                           verbose=verbose,
                                                           **kwargs)
                ctrls[gate.ctrl_y] = y[gate.ctrl_y].values
                if len(ctrls[gate.ctrl_x]) != len(ctrls[gate.ctrl_y]):
                    min_ = min([x.shape[0] for x in ctrls.values()])
                    ctrls = {k: np.random.choice(v, min_) for k, v in ctrls.items()}
            ctrls = pd.DataFrame(ctrls)
            return parent, ctrls
        return parent, None

    def _load_gate_dataframes_boolean(self,
                                      gate: BooleanGate):
        """
        Load Population dataframes for BooleanGate

        Parameters
        ----------
        gate: BooleanGate

        Returns
        -------
        List
            List of DataFrames
        """
        parent_data = self.filegroup.load_population_df(population=gate.parent,
                                                        transform=None,
                                                        label_downstream_affiliations=False)
        return [self.filegroup.load_population_df(population=pop,
                                                  transform=None,
                                                  label_downstream_affiliations=False)
                for pop in gate.populations], parent_data

    def apply_gate(self,
                   gate: Union[str, Gate, ThresholdGate, BooleanGate, PolygonGate, EllipseGate],
                   plot: bool = True,
                   verbose: bool = True,
                   add_to_strategy: bool = True,
                   create_plot_kwargs: Optional[Dict] = None,
                   plot_gate_kwargs: Optional[Dict] = None,
                   hyperparam_search: bool = True,
                   fda_norm: bool = False,
                   overwrite_method_kwargs: Optional[Dict] = None):
        """
        Apply a gate to the associated FileGroup. The gate must be previously defined;
        children associated and labeled. Either a Gate object can be provided or the name
        of an existing gate saved to this GatingStrategy.

        Parameters
        ----------
        gate: str or Gate or ThresholdGate or BooleanGate or PolygonGate or EllipseGate
            Name of an existing Gate or a Gate object
        plot: bool (default=True)
            If True, returns a Matplotlib.Axes object of plotted gate
        verbose: bool (default=True)
            If True, print gating statistics to stdout and provide feedback
        add_to_strategy: bool (default=True)
            If True, append the Gate to the GatingStrategy
        create_plot_kwargs: dict (optional)
            Additional arguments passed to CreatePlot
        plot_gate_kwargs: dict (optional)
            Additional arguments passed to plot_gate call of CreatePlot
        hyperparam_search: bool (default=True)
            If True and hyperparameter grid has been defined for the chosen gate,
            then hyperparameter search is performed to find the optimal fit for the
            newly encountered data.
        fda_norm: bool (default=False)
            Perform normalisation if defined for Gate
        overwrite_method_kwargs: dict, optional
            If a dictionary is provided (and hyperparameter search isn't defined for this gate)
            then method parameters are overwritten with these new parameters.

        Returns
        -------
        Matplotlib.Axes or None

        Raises
        ------
        AssertionError
            If control gating defined for a Gate other than a ThresholdGate

        DuplicateGateError
            If the Gate already exists and there add_to_strategy is True
        """
        if isinstance(gate, str):
            gate = self.get_gate(gate=gate)
            add_to_strategy = False

        if add_to_strategy:
            if gate.gate_name in self.list_gates():
                raise DuplicateGateError(f"Gate with name {gate.gate_name} already exists. "
                                         f"To continue set add_to_strategy to False")

        create_plot_kwargs = create_plot_kwargs or {}
        plot_gate_kwargs = plot_gate_kwargs or {}
        if isinstance(gate, BooleanGate):
            ctrl_parent_data = None
            data, parent_data = self._load_gate_dataframes_boolean(gate=gate)
        else:
            data, ctrl_parent_data = self._load_gate_dataframes(gate=gate, fda_norm=fda_norm, verbose=verbose)
            parent_data = data
        original_method_kwargs = gate.method_kwargs.copy()

        if overwrite_method_kwargs is not None:
            gate.method_kwargs = overwrite_method_kwargs
        if gate.ctrl_x is not None:
            assert isinstance(gate, ThresholdGate), "Control gate only supported for ThresholdGate"
            populations = gate.fit_predict(data=data, ctrl_data=ctrl_parent_data)
        elif gate.gate_name in self.hyperparameter_search.keys() and hyperparam_search and not isinstance(gate,
                                                                                                          BooleanGate):
            populations = hyperparameter_gate(gate=gate,
                                              grid=self.hyperparameter_search.get(gate.gate_name).get("grid"),
                                              cost=self.hyperparameter_search.get(gate.gate_name).get("cost"),
                                              parent=data,
                                              verbose=verbose)
        else:
            populations = gate.fit_predict(data=data)
        for p in populations:
            self.filegroup.add_population(population=p)
        if verbose:
            gate_stats(gate=gate, parent_data=parent_data, populations=populations)
        if add_to_strategy:
            self.gates.append(gate)
        if plot:
            plot = FlowPlot(**create_plot_kwargs)
            return plot.plot_population_geoms(parent=parent_data,
                                              children=populations,
                                              **plot_gate_kwargs)
        gate.method_kwargs = original_method_kwargs
        return None

    def apply_all(self,
                  verbose: bool = True,
                  fda_norm: bool = False,
                  hyperparam_search: bool = False):
        """
        Apply all the gates associated to this GatingStrategy

        Parameters
        ----------
        verbose: bool (default=True)
            If True, print feedback to stdout
        fda_norm: bool (default=False)
             Perform normalisation if defined for Gate
        hyperparam_search: bool (default=False)
            If True and hyperparameter grid has been defined for the chosen gate,
            then hyperparameter search is performed to find the optimal fit for the
            newly encountered data.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            No gates to apply

        DuplicatePopulationError
            If there is an attempt to create a population that already exists

        InsufficientEventsError
            Attempt to apply a gate where parent population has insufficient or no events

        OverflowError
            Failure to identify populations required to apply all gates downstream of root population
        """
        feedback = vprint(verbose)
        populations_created = [[c.name for c in g.children] for g in self.gates]
        populations_created = [x for sl in populations_created for x in sl]
        assert len(self.gates) > 0, "No gates to apply"
        err = "One or more of the populations generated from this gating strategy are already " \
              "presented in the population tree"
        if not all([x not in self.list_populations() for x in populations_created]):
            raise DuplicatePopulationError(err)
        gates_to_apply = list(self.gates)
        i = 0
        iteration_limit = len(gates_to_apply) * 100
        feedback("=====================================================")
        while len(gates_to_apply) > 0:
            if i >= len(gates_to_apply):
                i = 0
            gate = gates_to_apply[i]
            if gate.parent in self.list_populations():
                if self.filegroup.population_stats(gate.parent).get("n") <= 3:
                    raise InsufficientEventsError(f"Insufficient events in parent population {gate.parent}")
                feedback(f"------ Applying {gate.gate_name} ------")
                self.apply_gate(gate=gate,
                                plot=False,
                                verbose=verbose,
                                add_to_strategy=False,
                                fda_norm=fda_norm,
                                hyperparam_search=hyperparam_search)
                feedback("----------------------------------------")
                gates_to_apply = [g for g in gates_to_apply if g.gate_name != gate.gate_name]
            i += 1
            iteration_limit -= 1
            if iteration_limit == 0:
                raise OverflowError("Maximum number of iterations reached. This means that one or more parent "
                                    "populations are not being identified.")

    def apply_to_experiment(self,
                            experiment: Experiment,
                            fda_norm: bool = False,
                            hyperparam_search: bool = False,
                            plots_path: Optional[str] = None,
                            sample_ids: Union[list, None] = None,
                            verbose: bool = True):
        """
        Apply all the gates associated to this GatingStrategy to each FileGroup of
        an Experiment in sequence.

        Parameters
        ----------
        experiment: Experiment
            Experiment to apply GatingStrategy to
        fda_norm: bool (default=False)
             Perform normalisation if defined for Gate
        hyperparam_search: bool (default=False)
            If True and hyperparameter grid has been defined for the chosen gate,
            then hyperparameter search is performed to find the optimal fit for the
            newly encountered data.
        plots_path: str, optional
            If provided, a grid of plots will be generated for each sample showing
            each gate in sequence. Plots are saved to the specified path with each sample
            generating a png image with the filename corresponding to the sample ID
        sample_ids: list, optional
            If provided, only samples in this list have gates applied
        verbose: bool (default=True)
            Whether to provide a progress bar

        Returns
        -------
        None
        """
        logger.info(f" -- Gating {experiment.experiment_id} using {self.name} strategy --")

        sample_ids = sample_ids or experiment.list_samples()
        if plots_path is not None:
            assert os.path.isdir(plots_path), "Invalid plots_path, directory does not exist"
        for s in progress_bar(sample_ids, verbose=verbose):
            self.load_data(experiment=experiment, sample_id=s)
            try:
                self.apply_all(verbose=False,
                               fda_norm=fda_norm,
                               hyperparam_search=hyperparam_search)
                self.save(save_strategy=False, save_filegroup=True)
                logger.info(f"{s} - gated successfully!")
                if plots_path is not None:
                    fig = self.plot_all_gates()
                    fig.savefig(f"{plots_path}/{s}.png", facecolor="white", dpi=100)
                    plt.close(fig)
                    logger.info(f"{s} - gates plotted to {plots_path}")
            except DuplicatePopulationError as e:
                logger.error(f"{s} - {str(e)}")
            except InsufficientEventsError as e:
                logger.error(f"{s} - {str(e)}")
            except AssertionError as e:
                logger.error(f"{s} - {str(e)}")
            except ValueError as e:
                logger.error(f"{s} - {str(e)}")
            except OverflowError as e:
                logger.error(f"{s} - {str(e)}")
        logger.info(f" -- {experiment.experiment_id} complete --")

    def plot_all_gates(self):
        """
        Generates a grid of plots, one for each gate.

        Returns
        -------
        Matplotlib.Figure
        """
        n = len(self.gates)
        cols = 2
        rows = int(math.ceil(n / cols))
        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure(figsize=(12, 5 * rows))
        for i in range(n):
            ax = fig.add_subplot(gs[i])
            self.plot_gate(gate=self.gates[i].gate_name,
                           create_plot_kwargs={"ax": ax,
                                               "downsample": 0.25})
            ax.set_title(self.gates[i].gate_name)
        fig.tight_layout()
        return fig

    def delete_gate(self,
                    gate_name: str):
        """
        Remove a gate from this GatingStrategy. Note: populations generated from this
        gate will not be deleted. These populations must be deleted separately by calling
        the 'delete_population' method.

        Parameters
        ----------
        gate_name: str
            Name of the gate for removal
        Returns
        -------
        None
        """
        self.gates = [g for g in self.gates if g.gate_name != gate_name]

    def delete_populations(self,
                           populations: Union[str, list]):
        """
        Delete given populations. Populations downstream from delete population(s) will
        also be removed.

        Parameters
        ----------
        populations: list or str
            Either a list of populations (list of strings) to remove or a single population as a string.
            If a value of "all" is given, all populations are dropped.

        Returns
        -------
        None
        """
        self.filegroup.delete_populations(populations=populations)

    def plot_gate(self,
                  gate: str,
                  create_plot_kwargs: Optional[Dict] = None,
                  **kwargs):
        """
        Plot a gate. Must provide the name of a Gate currently associated to this GatingStrategy.
        This will plot the parent population this gate acts on along with the geometries
        that define the child populations the gate generates.

        Parameters
        ----------
        gate: str or Gate
        create_plot_kwargs: dict
            Keyword arguments for CreatePlot object. See cytopy.plotting.CreatePlot for details.
        kwargs:
            Keyword arguments for plot_gate call.
            See cytopy.plotting.CreatePlot.plot_population_geom for details.

        Returns
        -------
        Matplotlib.Axes

        Raises
        ------
        AssertionError
            Invalid gate provided; either it does not exist or has not been applied to the associated
            FileGroup
        """
        create_plot_kwargs = create_plot_kwargs or {}
        assert isinstance(gate, str), "Provide the name of an existing Gate in this GatingStrategy"
        assert gate in self.list_gates(), \
            f"Gate {gate} not recognised. Have you applied it and added it to the strategy?"
        gate = self.get_gate(gate=gate)
        fda_norm = gate.gate_name in self.normalisation.keys()
        parent, _ = self._load_gate_dataframes(gate=gate, fda_norm=fda_norm, ctrl=False)
        plotting = FlowPlot(**create_plot_kwargs)
        return plotting.plot_population_geoms(parent=parent,
                                              children=[self.filegroup.get_population(c.name)
                                                        for c in gate.children],
                                              **kwargs)

    def plot_backgate(self,
                      parent: str,
                      overlay: list,
                      x: str,
                      y: Optional[str] = None,
                      create_plot_kwargs: Optional[Dict] = None,
                      **backgate_kwargs):
        """
        Given some population as the backdrop (parent) and a list of one or more
        populations that occur downstream of the parent (overlay), plot the downstream
        populations as scatter plots over the top of the parent.

        Parameters
        ----------
        parent: str
        overlay: list
        x: str
        y: str
        create_plot_kwargs
            Additional keyword arguments passed to cytopy.flow.plotting.CreatePlot
        backgate_kwargs
            Additional keyword arguments passed to cytopy.flow.plotting.CreatePlot.backgate

        Returns
        -------
        Matplotlib.Axes

        Raises
        ------
        MissingPopulationError
            Parent population or population(s) in overlay are missing

        AssertionError
            One or more of the populations given in overlay is not downstream of the parent
        """
        if parent not in self.list_populations():
            raise MissingPopulationError("Parent population does not exist")
        if not all([x in self.list_populations() for x in overlay]):
            raise MissingPopulationError("One or more given populations could not be found")
        downstream = self.filegroup.list_downstream_populations(population=parent)
        assert all([x in downstream for x in overlay]), \
            "One or more of the given populations is not downstream of the given parent"
        create_plot_kwargs = create_plot_kwargs or {}
        plotting = FlowPlot(**create_plot_kwargs)
        parent = self.filegroup.load_population_df(population=parent,
                                                   transform=None,
                                                   label_downstream_affiliations=False)
        children = {x: self.filegroup.load_population_df(population=x,
                                                         transform=None,
                                                         label_downstream_affiliations=False)
                    for x in overlay}
        return plotting.backgate(parent=parent,
                                 children=children,
                                 x=x,
                                 y=y,
                                 **backgate_kwargs)

    def plot_population(self,
                        population: str,
                        x: str,
                        y: Optional[str] = None,
                        transform_x: Optional[str] = "logicle",
                        transform_y: Optional[str] = "logicle",
                        create_plot_kwargs: Optional[Dict] = None,
                        **plot_kwargs):
        """
        Plot an existing population in the associate FileGroup.

        Parameters
        ----------
        population: str
        x: str
        y: str (optional)
        transform_x: str (optional; default="logicle")
        transform_y: str (optional; default="logicle")
        create_plot_kwargs:
            Additional keyword arguments passed to cytopy.flow.plotting.CreatePlot
        plot_kwargs
            Additional keyword arguments passed to cytopy.flow.plotting.CreatePlot.plot

        Returns
        -------
        Matplotlib.Axes

        Raises
        ------
        MissingPopulationError
            Population is missing
        """
        if population not in self.list_populations():
            raise MissingPopulationError(f"{population} does not exist")
        data = self.filegroup.load_population_df(population=population,
                                                 transform=None,
                                                 label_downstream_affiliations=False)
        create_plot_kwargs = create_plot_kwargs or {}
        plotting = FlowPlot(transform_x=transform_x,
                            transform_y=transform_y,
                            **create_plot_kwargs)
        return plotting.plot(data=data, x=x, y=y, **plot_kwargs)

    def print_population_tree(self, **kwargs):
        """
        Print the population tree to stdout.
        Wraps cytopy.data.fcs.FileGroup.print_population_tree

        Parameters
        ----------
        kwargs
            See keyword arguments for cytopy.data.fcs.FileGroup.print_population_tree

        Returns
        -------
        None
        """
        self.filegroup.print_population_tree(**kwargs)

    def edit_gate(self,
                  gate_name: str,
                  x_threshold: float or None = None,
                  y_threshold: float or None = None,
                  coords: dict or None = None):
        """
        Edit an existing gate (i.e. the polygon or threshold shape that generates the resulting
        populations). The altered geometry will be applied to the parent population resulting
        this gate acts upon, resulting in new data. Populations downstream of this edit will
        also be effected but gates will not adapt dynamically, instead the static results of
        gating algorithms will still apply, but to a new dataset. For this reason, gates
        should be checked (similar to the effects of moving a gate in FlowJo).

        Parameters
        ----------
        gate_name: str
        x_threshold: float (optional)
            Required for threshold geometries
        y_threshold: float (optional)
            Required for 2D threshold geometries
        coords: dict (optional)
            Required for Polygon or Ellipse gate. Keys should be populations generated by the gate and
            values a numpy array like: [x_coordinates, y_coordinates].
        Returns
        -------
        None

        Raises
        ------
        AssertionError
            Attempting to edit a gate that has not been applied yet

        MissingPopulationError
            Population missing from coords

        ValueError
            Updated threshold or coords are invalid
        """
        gate = self.get_gate(gate=gate_name)
        err = "Cannot edit a gate that has not been applied; gate children not present in population tree."
        assert all([x in self.filegroup.tree.keys() for x in [c.name for c in gate.children]]), err
        transforms, transform_kwargs = gate.transform_info()
        parent = self.filegroup.load_population_df(population=gate.parent,
                                                   transform=transforms,
                                                   transform_kwargs=transform_kwargs)

        for child in gate.children:
            pop = self.filegroup.get_population(population_name=child.name)
            if isinstance(pop.geom, ThresholdGeom):
                if x_threshold is None:
                    raise ValueError("For threshold geometry, please provide x_threshold")
                xt = apply_transform(pd.DataFrame({"x": [x_threshold]}),
                                     features=["x"],
                                     method=transforms.get(gate.x),
                                     **transform_kwargs.get(gate.x)).x.values[0]
                yt = None
                if pop.geom.y_threshold is not None:
                    if y_threshold is None:
                        raise ValueError("For 2D threshold geometry, please provide y_threshold")
                    yt = apply_transform(pd.DataFrame({"x": [y_threshold]}),
                                         features=["x"],
                                         method=transforms.get(gate.y),
                                         **transform_kwargs.get(gate.y)).x.values[0]
                self.filegroup.update_population(update_threshold(population=pop,
                                                                  parent_data=parent,
                                                                  x_threshold=xt,
                                                                  y_threshold=yt))
            elif isinstance(pop.geom, PolygonGeom):
                if pop.population_name not in coords.keys():
                    raise MissingPopulationError(f"{pop.population_name} missing from coords")
                if not len(coords.get(pop.population_name)) == 2:
                    raise ValueError("coords values should be array of shape (2, n) where "
                                     "n is the desired number of coordinates")
                x_values, y_values = coords.get(pop.population_name)
                xc = apply_transform(pd.DataFrame({"x": [x_values]}),
                                     features=["x"],
                                     method=transforms.get(gate.x),
                                     **transform_kwargs.get(gate.x)).x.values[0]
                yc = apply_transform(pd.DataFrame({"y": [y_values]}),
                                     features=["y"],
                                     method=transforms.get(gate.y),
                                     **transform_kwargs.get(gate.y)).y.values[0]
                self.filegroup.update_population(update_polygon(population=pop,
                                                                parent_data=parent,
                                                                x_values=xc,
                                                                y_values=yc))
            self._edit_downstream_effects(population_name=child.name)

    def _edit_downstream_effects(self,
                                 population_name: str):
        """
        Echos the downstream effects of an edited gate by iterating over the Population
        dependencies and reapplying their geometries to the modified data. Should be
        called after 'edit_population'.

        Parameters
        ----------
        population_name: str

        Returns
        -------
        None
        """
        downstream_populations = self.filegroup.list_downstream_populations(population=population_name)
        for pop in downstream_populations:
            pop = self.filegroup.get_population(pop)
            transforms = {k: v for k, v in zip([pop.geom.x, pop.geom.y],
                                               [pop.geom.transform_x, pop.geom.transform_y])
                          if k is not None}
            parent = self.filegroup.load_population_df(population=pop.parent,
                                                       transform=transforms)
            if isinstance(pop.geom, ThresholdGeom):
                self.filegroup.update_population(update_threshold(population=pop,
                                                                  parent_data=parent,
                                                                  x_threshold=pop.geom.x_threshold,
                                                                  y_threshold=pop.geom.y_threshold))
            elif isinstance(pop.geom, PolygonGeom):
                self.filegroup.update_population(update_polygon(population=pop,
                                                                parent_data=parent,
                                                                x_values=pop.geom.x_values,
                                                                y_values=pop.geom.y_values))

    def save(self,
             save_strategy: bool = True,
             save_filegroup: bool = True,
             *args,
             **kwargs):
        """
        Save GatingStrategy and the populations generated for the associated
        FileGroup.

        Parameters
        ----------
        save_filegroup: bool (default=True)
        save_strategy: bool (default=True)
        args:
            Positional arguments for mongoengine.document.save call
        kwargs:
            Keyword arguments for mongoengine.document.save call

        Returns
        -------
        None
        """
        if save_strategy:
            for g in self.gates:
                g.save()
            self.last_edit = datetime.now()
            super().save(*args, **kwargs)
        if save_filegroup:
            if self.name not in self.filegroup.gating_strategy:
                self.filegroup.gating_strategy.append(self.name)
            if self.filegroup is not None:
                self.filegroup.save()

    def delete(self,
               delete_gates: bool = True,
               remove_associations: bool = True,
               *args, **kwargs):
        """
        Delete gating strategy. If delete_gates is True, then associated Gate objects will
        also be deleted. If remove_associations is True, then populations generated from
        this gating strategy will also be deleted.

        Parameters
        ----------
        delete_gates: bool (default=True)
        remove_associations: (default=True)
        args:
            Positional arguments for mongoengine.document.delete call
        kwargs:
            Keyword arguments for mongoengine.document.delete call

        Returns
        -------

        """
        super().delete(*args, **kwargs)
        populations = [[c.name for c in g.children] for g in self.gates]
        populations = list(set([x for sl in populations for x in sl]))
        if delete_gates:
            logger.info("Deleting gates...")
            for g in self.gates:
                g.delete()
        if remove_associations:
            logger.info("Deleting associated populations in FileGroups...")
            for f in progress_bar(FileGroup.objects(), verbose=self.verbose):
                try:
                    if self.name in f.gating_strategy:
                        f.gating_strategy = [gs for gs in f.gating_strategy if gs != self.name]
                        f.delete_populations(populations=populations)
                        f.save()
                except ValueError as e:
                    logger.warning(f"Could not delete associations in {f.primary_id}: {e}")

        logger.info(f"{self.name} successfully deleted.")
