import logging
import os
from collections import defaultdict
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import mongoengine
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mongoengine import DoesNotExist

from cytopy import Experiment
from cytopy.data.errors import DuplicatePopulationError
from cytopy.data.errors import GateError
from cytopy.data.errors import InsufficientEventsError
from cytopy.data.errors import MissingPopulationError
from cytopy.data.fcs import copy_populations_to_controls_using_geoms
from cytopy.data.fcs import FileGroup
from cytopy.data.population import ThresholdGeom
from cytopy.feedback import progress_bar
from cytopy.gating.base import Gate
from cytopy.gating.polygon import update_polygon
from cytopy.gating.threshold import update_threshold
from cytopy.plotting.cyto import cyto_plot
from cytopy.plotting.cyto import overlay
from cytopy.plotting.cyto import plot_gate
from cytopy.plotting.general import build_plot_grid

logger = logging.getLogger(__name__)


class GatingStrategy(mongoengine.Document):
    name = mongoengine.StringField(required=True, unique=True)
    gates = mongoengine.ListField(mongoengine.ReferenceField(Gate, reverse_delete_rule=mongoengine.PULL))
    hyperparameter_search = mongoengine.DictField()
    meta = {"db_alias": "core", "collection": "gating_strategy"}

    def __init__(self, *args, **values):
        super().__init__(*args, **values)
        self.filegroup = None
        self.data_source = "primary"

    def load_data(self, experiment: Experiment, sample_id: str):
        self.filegroup = experiment.get_sample(sample_id=sample_id)

    def add_gate(self, gate: Gate):
        if gate.gate_name in [g.name for g in self.gates]:
            raise ValueError(f"Gate {gate.gate_name} already exists.")
        self.gates.append(gate)

    def delete_gate(self, name: str, delete_populations: bool = True):
        gate = self.get_gate(gate=name)
        if delete_populations:
            populations = [c.name for c in gate.children]
            self.delete_populations(populations=populations)
        self.gates = [g for g in self.gates if g.gate_name != name]
        gate.delete()

    def list_gates(self):
        return [g.gate_name for g in self.gates]

    def get_gate(self, gate: str):
        try:
            return [g for g in self.gates if g.gate_name == gate][0]
        except IndexError:
            raise DoesNotExist(f"Gate {gate} does not exists.")

    def get_population_gate(self, population_name: str):
        gate = [g for g in self.gates if population_name in [c.name for c in g.children]]
        if len(gate) == 0:
            raise DoesNotExist(f"No corresponding gate found for population '{population_name}'")
        return gate[0]

    def list_populations(self, **kwargs):
        return self.filegroup.list_populations(**kwargs)

    def delete_populations(self, populations: List[str]):
        self.filegroup.delete_populations(populations=populations, data_source=self.data_source)

    def print_population_tree(self, **kwargs):
        return self.filegroup.print_population_tree(data_source=self.data_source, **kwargs)

    def _population_data(self, population_name: str):
        return self.filegroup.load_population_df(
            population=population_name, transform=None, data_source=self.data_source
        )

    def propagate_to_control(self, ctrl: str, flag: float = 0.25):
        return copy_populations_to_controls_using_geoms(filegroup=self.filegroup, ctrl=ctrl, flag=flag)

    def add_hyperparameter_search(self, gate: str, params: Dict):
        gate = self.get_gate(gate)
        for key in params.keys():
            if gate.model is not None:
                if key in gate.model.get_params().keys():
                    continue
            if key in dict(gate.to_mongo()).keys():
                continue
            raise KeyError(f"{key} is not a valid parameter for the requested Gate.")
        self.hyperparameter_search[gate.gate_name] = params
        return self

    def plot_gate(self, gate: str, plot_training_data: bool = False, **plot_kwargs):
        fg = self.filegroup if plot_training_data else None
        return plot_gate(gate=self.get_gate(gate), filegroup=fg, data_source=self.data_source, **plot_kwargs)

    def plot_all_gates(
        self,
        plot_training_data: bool = False,
        col_wrap: int = 2,
        figure_kwargs: Optional[Dict] = None,
        plot_kwargs: Optional[Dict] = None,
    ):
        figure_kwargs = figure_kwargs or {}
        plot_kwargs = plot_kwargs or {}
        fig, axes = build_plot_grid(n=len(self.gates), col_wrap=col_wrap, **figure_kwargs)
        for i, gate in enumerate(self.gates):
            self.plot_gate(
                gate=gate.gate_name,
                plot_training_data=plot_training_data,
                ax=axes[i],
                **plot_kwargs.get(gate.gate_name, {}),
            )
            axes[i].set_title(gate.gate_name)
        fig.tight_layout()
        return fig

    def plot_population(
        self,
        population_name: str,
        x: str,
        y: Optional[str] = None,
        transform_x: str = "asinh",
        transform_y: str = "asinh",
        **plot_kwargs,
    ):
        return cyto_plot(
            data=self._population_data(population_name=population_name),
            x=x,
            y=y,
            transform_x=transform_x,
            transform_y=transform_y,
            **plot_kwargs,
        )

    def plot_overlay(
        self,
        x: str,
        y: str,
        background_population: str,
        overlay_populations: List[str],
        transform_x: str = "asinh",
        transform_y: str = "asinh",
        legend_kwargs: Optional[Dict] = None,
        **plot_kwargs,
    ):
        background_population = self._population_data(population_name=background_population)
        overlay_populations = {p: self._population_data(population_name=p) for p in overlay_populations}
        return overlay(
            x=x,
            y=y,
            background_data=background_population,
            overlay_data=overlay_populations,
            transform_x=transform_x,
            transform_y=transform_y,
            legend_kwargs=legend_kwargs,
            **plot_kwargs,
        )

    def _gate_children_exist(self, gate: Gate):
        try:
            for child in gate.children:
                assert child.name in self.filegroup.tree.keys()
        except AssertionError:
            raise ValueError(
                "Cannot edit a gate that has not been applied; gate children not present in population tree."
            )
        return gate

    def edit_threshold_populations(self, gate: str, x_threshold: float, y_threshold: Optional[float] = None):
        gate = self._gate_children_exist(gate=self.get_gate(gate=gate))
        parent = self._population_data(population_name=gate.parent)
        parent = gate.preprocess(data=parent, transform=True)
        thresholds = pd.DataFrame({gate.x: [x_threshold], gate.y: [y_threshold]})
        thresholds = gate.transform(data=thresholds)
        for child in gate.children:
            pop = self.filegroup.get_population(population_name=child.name, data_source=self.data_source)
            self.filegroup.update_population(
                update_threshold(
                    population=pop,
                    parent_data=parent,
                    x_threshold=thresholds[gate.x].values[0],
                    y_threshold=thresholds[gate.y].values[0],
                )
            )
            self._edit_downstream_effects(population_name=child.name)

    def edit_polygon_populations(self, gate: str, coords: Dict[str, Iterable[float]]):
        gate = self._gate_children_exist(gate=self.get_gate(gate=gate))
        parent = self._population_data(population_name=gate.parent)
        parent = gate.preprocess(data=parent, transform=True)
        for child in gate.children:
            pop = self.filegroup.get_population(population_name=child.name, data_source=self.data_source)
            try:
                xy = np.array(coords[pop.population_name])
                assert xy.shape[1] == 2
                xy = gate.transform(data=pd.DataFrame(xy, columns=[gate.x, gate.y])).values
                self.filegroup.update_population(
                    update_polygon(population=pop, parent_data=parent, x_values=xy[:, 0], y_values=xy[:, 1])
                )
                self._edit_downstream_effects(population_name=child.name)
            except KeyError:
                raise MissingPopulationError(f"{pop.population_name} missing from coords")
            except AssertionError:
                raise GateError("coords should be of shape (2, n) where n id the desired number of coordinates")

    def _edit_downstream_effects(self, population_name: str):
        downstream_populations = self.filegroup.list_downstream_populations(
            population=population_name, data_source=self.data_source
        )
        for pop in downstream_populations:
            gate = self.get_population_gate(population_name=pop.population_name)
            parent = gate.preprocess(self._population_data(population_name=gate.parent), transform=True)
            if isinstance(pop.geom, ThresholdGeom):
                self.filegroup.update_population(
                    update_threshold(
                        population=pop,
                        parent_data=parent,
                        x_threshold=pop.geom.x_threshold,
                        y_threshold=pop.geom.y_threshold,
                    )
                )
            else:
                self.filegroup.update_population(
                    update_polygon(
                        population=pop, parent_data=parent, x_values=pop.geom.x_values, y_values=pop.geom.y_values
                    )
                )

    def preview(self, gate: Union[str, Gate], **plot_kwargs):
        if isinstance(gate, str):
            gate = self.get_gate(gate)
        gate.train(data=self._population_data(population_name=gate.parent), transform=True)
        return plot_gate(gate=gate, **plot_kwargs)

    def apply(
        self,
        gate: Union[str, Gate],
        plot: bool = True,
        njobs: int = -1,
        overwrite_kwargs: Optional[Dict] = None,
        print_stats: bool = True,
        **plot_kwargs,
    ):
        overwrite_kwargs = overwrite_kwargs or {}
        if isinstance(gate, str):
            gate = self.get_gate(gate)
        if gate.gate_name in self.hyperparameter_search.keys():
            populations = gate.predict_with_hyperparameter_search(
                data=self._population_data(population_name=gate.parent),
                parameter_grid=self.hyperparameter_search[gate.gate_name],
                transform=True,
                njobs=njobs,
            )
        else:
            populations = gate.predict(
                data=self._population_data(population_name=gate.parent), transform=True, **overwrite_kwargs
            )
        for pop in populations:
            self.filegroup.add_population(population=pop)
        if print_stats:
            print(f"----- {gate.gate_name} -----")
            parent_n = self.filegroup.get_population(population_name=gate.parent, data_source=self.data_source).n
            print(f"Parent ({gate.parent}) n: {parent_n}")
            for p in populations:
                print(f"...child {p.population_name} n: {p.n}; {p.n / parent_n * 100}% of parent")
            print("------------------------")
        if plot:
            return plot_gate(gate=gate, filegroup=self.filegroup, data_source=self.data_source, **plot_kwargs)
        return self

    def apply_all(self) -> pd.DataFrame:
        assert len(self.gates) > 0, "No gates to apply"
        populations_created = [[c.name for c in g.children] for g in self.gates]
        populations_created = [x for sl in populations_created for x in sl]
        if not all([x not in self.list_populations() for x in populations_created]):
            raise DuplicatePopulationError("One or more populations created by this gating strategy already exist!")
        gates_to_apply = list(self.gates)
        i = 0
        iteration_limit = len(gates_to_apply) * 100
        stats = defaultdict(list)
        root_n = self.filegroup.get_population(population_name="root", data_source=self.data_source)
        logger.info(f"Applying template to {self.filegroup.primary_id}")
        while len(gates_to_apply) > 0:
            if i >= len(gates_to_apply):
                i = 0
            gate = gates_to_apply[i]
            if gate.parent in self.list_populations():
                if self.filegroup.population_stats(gate.parent).get("n") <= 3:
                    raise InsufficientEventsError(
                        population_id=gate.parent,
                        filegroup_id=self.filegroup.primary_id,
                    )
                logger.info(f"...applying gate '{gate.gate_name}'")
                self.apply(gate=gate, plot=False, print_stats=False)
                for child in gate.children:
                    n = self.filegroup.get_population(population_name=child.name, data_source=self.data_source).n
                    parent_n = self.filegroup.get_population(
                        population_name=gate.parent, data_source=self.data_source
                    ).n
                    stats["Gate"].append(gate.gate_name)
                    stats["Population"].append(child.name)
                    stats["% of parent"].append(n / parent_n * 100)
                    stats["% of root"].append(n / root_n * 100)
                gates_to_apply = [g for g in gates_to_apply if g.gate_name != gate.gate_name]
            i += 1
            iteration_limit -= 1
            if iteration_limit == 0:
                raise OverflowError(
                    "Maximum number of iterations reached. This means that one or more parent "
                    "populations are not being identified."
                )
        return pd.DataFrame(stats)

    def apply_to_experiment(
        self,
        experiment: Experiment,
        plots_path: Optional[str] = None,
        sample_ids: Union[list, None] = None,
        verbose: bool = True,
    ):
        """
        Apply all the gates associated to this GatingStrategy to each FileGroup of
        an Experiment in sequence.

        Parameters
        ----------
        experiment: Experiment
            Experiment to apply GatingStrategy to
        plots_path: str, optional
            If provided, a grid of plots will be generated for each sample showing
            each gate in sequence. Plots are saved to the specified path with each sample
            generating a png image with the filename corresponding to the sample ID
        sample_ids: list, optional
            If provided, only samples in this list have gates applied

        Returns
        -------
        None
        """
        logger.info(f" -- Gating {experiment.experiment_id} using {self.name} strategy --")
        stats = list()
        sample_ids = sample_ids or experiment.list_samples()
        if plots_path is not None:
            assert os.path.isdir(plots_path), "Invalid plots_path, directory does not exist"
        for s in progress_bar(sample_ids, verbose=verbose):
            self.load_data(experiment=experiment, sample_id=s)
            try:
                sample_stats = self.apply_all()
                sample_stats["sample_id"] = s
                stats.append(sample_stats)
                self.save(save_strategy=False, save_filegroup=True)
                if plots_path is not None:
                    fig = self.plot_all_gates()
                    fig.savefig(f"{plots_path}/{s}.png", facecolor="white", dpi=100)
                    plt.close(fig)
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
        return pd.concat(stats).set_index("sample_id")

    def save(self, save_strategy: bool = True, save_filegroup: bool = True, *args, **kwargs):
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
            super().save(*args, **kwargs)
        if save_filegroup:
            if self.name not in self.filegroup.gating_strategy:
                self.filegroup.gating_strategy.append(self.name)
            if self.filegroup is not None:
                self.filegroup.save()

    def delete(
        self,
        *args,
        **kwargs,
    ):
        """
        Delete gating strategy. If delete_gates is True, then associated Gate objects will
        also be deleted. If remove_associations is True, then populations generated from
        this gating strategy will also be deleted.

        Parameters
        ----------
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
        logger.info("Deleting associated gates...")
        for g in self.gates:
            g.delete()
        logger.info("Deleting associated populations in FileGroups...")
        for f in FileGroup.objects():
            try:
                if self.name in f.gating_strategy:
                    f.gating_strategy = [gs for gs in f.gating_strategy if gs != self.name]
                    f.delete_populations(populations=populations)
                    f.save()
            except ValueError as e:
                logger.warning(f"Could not delete associations in {f.primary_id}: {e}")
        logger.info(f"{self.name} successfully deleted.")
