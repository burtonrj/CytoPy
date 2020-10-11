from ..flow.plotting import CreatePlot
from ..feedback import progress_bar, vprint
from .gate import Gate, ThresholdGate, PolygonGate, EllipseGate
from .experiment import Experiment
from .fcs import FileGroup
from datetime import datetime
import mongoengine


class Action(mongoengine.EmbeddedDocument):
    action_name = mongoengine.StringField()
    method = mongoengine.StringField(choices=["merge", "subtract"])
    left = mongoengine.StringField()
    right = mongoengine.StringField()
    new_population_name = mongoengine.StringField()


class GatingStrategy(mongoengine.Document):
    """
    Document representation of a gating template; a gating template is a collection of gating objects
    that can be applied to multiple fcs files or an entire experiment in bulk

    Parameters
    -----------
    template_name: str, required
        unique identifier for template
    gates: EmbeddedDocumentList
        list of Gate documents; see Gate
    creation_date: DateTime
        date of creation
    last_edit: DateTime
        date of last edit
    flags: str, optional
        warnings associated to this gating template
    notes: str, optional
        free text comments
    """
    name = mongoengine.StringField(required=True, unique=True)
    gates = mongoengine.ListField(mongoengine.ReferenceField(Gate, reverse_delete_rule=mongoengine.PULL))
    actions = mongoengine.EmbeddedDocumentListField(Action)
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
        self.print = vprint(verbose=self.verbose)
        super().__init__(*args, **values)
        self._filegroup = None

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
        self._filegroup = experiment.get_sample(sample_id=sample_id)

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
        """
        assert self._filegroup is not None, "No FileGroup associated"
        return list(self._filegroup.list_populations())

    def _gate_exists(self,
                     gate: str):
        """
        Raises AssertionError if given gate does not exist

        Returns
        -------
        None
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
                     gate: str or Gate or ThresholdGate or PolygonGate or EllipseGate,
                     create_plot_kwargs: dict or None = None,
                     plot_gate_kwargs: dict or None = None):
        """
        Preview the results of some given Gate

        Parameters
        ----------
        gate: str or Gate or ThresholdGate or PolygonGate or EllipseGate
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
        parent_data = self._filegroup.load_population_df(population=gate.parent,
                                                         transform=None,
                                                         label_downstream_affiliations=False)
        gate.fit(data=parent_data)
        plot = CreatePlot(**create_plot_kwargs)
        return plot.plot_gate(gate=gate,
                              parent=parent_data,
                              **plot_gate_kwargs)

    def apply_gate(self,
                   gate: str or Gate or ThresholdGate or PolygonGate or EllipseGate,
                   plot: bool = True,
                   print_stats: bool = True,
                   create_plot_kwargs: dict or None = None,
                   plot_gate_kwargs: dict or None = None):
        """

        Parameters
        ----------
        gate: str or Gate or ThresholdGate or PolygonGate or EllipseGate
            Name of an existing Gate or a Gate object
        plot: bool (default=True)
            If True, returns a Matplotlib.Axes object of plotted gate
        print_stats: bool (default=True)
            If True, print gating statistics to stdout
        create_plot_kwargs: dict (optional)
            Additional arguments passed to CreatePlot
        plot_gate_kwargs: dict (optional)
            Additional arguments passed to plot_gate call of CreatePlot

        Returns
        -------
        Matplotlib.Axes or None
        """
        create_plot_kwargs = create_plot_kwargs or {}
        plot_gate_kwargs = plot_gate_kwargs or {}
        if isinstance(gate, str):
            gate = self.get_gate(gate=gate)
        parent_data = self._filegroup.load_population_df(population=gate.parent,
                                                         transform=None,
                                                         label_downstream_affiliations=False)
        populations = gate.fit_predict(data=parent_data)
        for p in populations:
            self._filegroup.add_population(population=p)
        if print_stats:
            print(f"----- {gate.gate_name} -----")
            parent_n = parent_data.shape[0]
            print(f"Parent ({gate.parent}) n: {parent_n}")
            for p in populations:
                print(f"...child {p.population_name} n: {p.n}; {p.n/parent_n*100}% of parent")
            print("------------------------")
        if plot:
            plot = CreatePlot(**create_plot_kwargs)
            return plot.plot_gate(gate=gate,
                                  parent=parent_data,
                                  **plot_gate_kwargs)
        return None

    def apply_all(self):
        pass

    def delete_gate(self):
        pass

    def edit_gate(self):
        pass

    def delete_population(self):
        pass

    def delete_action(self):
        pass

    def plot_gate(self):
        pass

    def plot_backgate(self):
        pass

    def plot_population(self):
        pass

    def print_population_tree(self):
        pass

    def population_stats(self):
        pass

    def estimate_ctrl_population(self):
        pass

    def merge_populations(self):
        pass

    def subtract_populations(self):
        pass

    def save(self, *args, **kwargs):
        for g in self.gates:
            g.save()
        super().save(*args, **kwargs)

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
        delete_gates
        remove_associations
        args
        kwargs

        Returns
        -------

        """
        if delete_gates:
            self.print("Deleting gates...")
            for g in self.gates:
                g.delete()
        if remove_associations:
            self.print("Deleting associated populations...")
            for f in progress_bar(FileGroup.objects(), verbose=self.verbose):
                if self.name in f.gating_strategy:
                    f.gating_strategy = [gs for gs in f.gating_strategy if gs != self.name]
                    f.populations = []
                    f.save()
        super().delete(*args, **kwargs)
        self.print(f"{self.name} successfully deleted.")
