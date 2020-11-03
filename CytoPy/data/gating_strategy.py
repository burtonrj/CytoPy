#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
In a traditional analysis, an immunologist would apply a 'gating strategy';
a series of 'gates' that separate single cell data into the populations of
interest. CytoPy provides autonomous gates (see CytoPy.data.gate) to
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

from ..flow.plotting import CreatePlot
from ..feedback import progress_bar, vprint
from .gate import Gate, ThresholdGate, PolygonGate, EllipseGate, ThresholdGeom, \
    PolygonGeom, update_polygon, update_threshold
from .experiment import Experiment
from .fcs import FileGroup
from datetime import datetime
import mongoengine

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


class Action(mongoengine.EmbeddedDocument):
    """
    An Action represents a process applied to the gates/populations in some gating strategy
    that is independent of the gates themselves. At the moment this includes merging populations
    or subtracting one population from another. These actions can appear in a gating strategy
    and will be applied to new data in an autonomous fashion.

    Attributes
    ----------
    action_name: str
        Name of the action
    method: str
        Should have a value of "merge" or "subtract"
    left: str
        The population to merge on or subtract from
    right: str
        The population to merge with or be subtracted from 'left'
    new_population_name: str
        Name of the new population generated from this action
    """
    action_name = mongoengine.StringField()
    method = mongoengine.StringField(choices=["merge", "subtract"])
    left = mongoengine.StringField()
    right = mongoengine.StringField()
    new_population_name = mongoengine.StringField()


class GatingStrategy(mongoengine.Document):
    """
    A GatingTemplate is synonymous to what an immunologist would classically consider
    a "gating template"; it is a collection of 'gates' (Gate objects, in the case of CytoPy)
    that can be applied to multiple fcs files or an entire experiment in bulk. A user defines
    a GatingTemplate using a single example from an experiment, uses the object to preview gates
    and label child populations, and when satisfied with the performance save the GatingStrategy
    to the database to be applied to the remaining samples in the Experiment.

    Attributes
    -----------
    template_name: str, required
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
        parent_data = self.filegroup.load_population_df(population=gate.parent,
                                                        transform=None,
                                                        label_downstream_affiliations=False)
        gate.fit(data=parent_data)
        plot = CreatePlot(**create_plot_kwargs)
        return plot.plot_gate_children(gate=gate,
                                       parent=parent_data,
                                       **plot_gate_kwargs)

    def apply_gate(self,
                   gate: str or Gate or ThresholdGate or PolygonGate or EllipseGate,
                   plot: bool = True,
                   print_stats: bool = True,
                   add_to_strategy: bool = True,
                   create_plot_kwargs: dict or None = None,
                   plot_gate_kwargs: dict or None = None):
        """
        Apply a gate to the associated FileGroup. The gate must be previously defined;
        children associated and labeled. Either a Gate object can be provided or the name
        of an existing gate saved to this GatingStrategy.

        Parameters
        ----------
        gate: str or Gate or ThresholdGate or PolygonGate or EllipseGate
            Name of an existing Gate or a Gate object
        plot: bool (default=True)
            If True, returns a Matplotlib.Axes object of plotted gate
        print_stats: bool (default=True)
            If True, print gating statistics to stdout
        add_to_strategy: bool (default=True)
            If True, append the Gate to the GatingStrategy
        create_plot_kwargs: dict (optional)
            Additional arguments passed to CreatePlot
        plot_gate_kwargs: dict (optional)
            Additional arguments passed to plot_gate call of CreatePlot

        Returns
        -------
        Matplotlib.Axes or None
        """
        if isinstance(gate, str):
            gate = self.get_gate(gate=gate)
            add_to_strategy = False
        if add_to_strategy:
            assert gate.gate_name not in self.list_gates(), \
                f"Gate with name {gate.gate_name} already exists. To continue set add_to_strategy to False"
        create_plot_kwargs = create_plot_kwargs or {}
        plot_gate_kwargs = plot_gate_kwargs or {}
        parent_data = self.filegroup.load_population_df(population=gate.parent,
                                                        transform=None,
                                                        label_downstream_affiliations=False)
        if gate.ctrl is None:
            populations = gate.fit_predict(data=parent_data)
        else:
            populations = self._control_gate(gate=gate)
        for p in populations:
            self.filegroup.add_population(population=p)
        if print_stats:
            print(f"----- {gate.gate_name} -----")
            parent_n = parent_data.shape[0]
            print(f"Parent ({gate.parent}) n: {parent_n}")
            for p in populations:
                print(f"...child {p.population_name} n: {p.n}; {p.n / parent_n * 100}% of parent")
            print("------------------------")
        if add_to_strategy:
            self.gates.append(gate)
        if plot:
            plot = CreatePlot(**create_plot_kwargs)
            return plot.plot_population_geoms(parent=parent_data,
                                              children=populations,
                                              **plot_gate_kwargs)
        return None

    def apply_all(self,
                  verbose: bool = True):
        """
        Apply all the gates associated to this GatingStrategy

        Parameters
        ----------
        verbose: bool (default=True)
            If True, print feedback to stdout

        Returns
        -------
        None
        """
        feedback = vprint(verbose)
        populations_created = [[c.name for c in g.children] for g in self.gates]
        populations_created = [x for sl in populations_created for x in sl]
        assert len(self.gates) > 0, "No gates to apply"
        err = "One or more of the populations generated from this gating strategy are already " \
              "presented in the population tree"
        assert all([x not in self.list_populations() for x in populations_created]), err
        gates_to_apply = list(self.gates)
        actions_to_apply = list(self.actions)
        i = 0
        iteration_limit = len(gates_to_apply) * 100
        feedback("=====================================================")
        while len(gates_to_apply) > 0:
            if i >= len(gates_to_apply):
                i = 0
            gate = gates_to_apply[i]
            if gate.parent in self.list_populations():
                feedback(f"------ Applying {gate.gate_name} ------")
                self.apply_gate(gate=gate,
                                plot=False,
                                print_stats=verbose,
                                add_to_strategy=False)
                feedback("----------------------------------------")
                gates_to_apply = [g for g in gates_to_apply if g.gate_name != gate.gate_name]
            actions_applied_this_loop = list()
            for a in actions_to_apply:
                if a.left in self.list_populations() and a.right in self.list_populations():
                    feedback(f"------ Applying {a.action_name} ------")
                    self.apply_action(action=a,
                                      print_stats=verbose,
                                      add_to_strategy=False)
                    feedback("----------------------------------------")
                    actions_applied_this_loop.append(a.action_name)
            actions_to_apply = [a for a in actions_to_apply
                                if a.action_name not in actions_applied_this_loop]
            i += 1
            iteration_limit -= 1
            assert iteration_limit > 0, "Maximum number of iterations reached. This means that one or more parent " \
                                        "populations are not being identified."

    def delete_actions(self,
                       action_name: str):
        """
        Delete an action associated to this GatingStrategy

        Parameters
        ===========
        action_name: str

        Returns
        -------
        None
        """
        self.actions = [a for a in self.actions if a.action_name != action_name]

    def apply_action(self,
                     action: Action or str,
                     print_stats: bool = True,
                     add_to_strategy: bool = True):
        """
        Apply an action, that is, a merge or subtraction:
            * Merge: merge two populations present in the current population tree.
            The merged population will have the combined index of both populations but
            will not inherit any clusters and will not be associated to any children
            downstream of either the left or right population. The population will be
            added to the tree as a descendant of the left populations parent
            * Subtraction: subtract the right population from the left population.
            The right population must either have the same parent as the left population
            or be downstream of the left population. The new population will descend from
            the same parent as the left population. The new population will have a
            PolygonGeom geom.

        Parameters
        ----------
        action: Action
        print_stats: bool (default=True)
            Print population statistics to stdout
        add_to_strategy: bool (default=True)
            Add action to this GatingStrategy
        Returns
        -------
        None
        """
        if isinstance(action, str):
            matching_action = [a for a in self.actions if a.action_name == action]
            assert len(matching_action) == 1, f"{action} does not exist"
            action = matching_action[0]
        assert action.method in ["merge", "subtract"], "Accepted methods are: merge, subtract"
        assert action.left in self.list_populations(), f"{action.left} does not exist"
        assert action.right in self.list_populations(), f"{action.right} does not exist"
        left = self.filegroup.get_population(action.left)
        right = self.filegroup.get_population(action.right)
        if action.method == "merge":
            self.filegroup.merge_populations(left=left,
                                             right=right,
                                             new_population_name=action.new_population_name)
        else:
            self.filegroup.subtract_populations(left=left,
                                                right=right,
                                                new_population_name=action.new_population_name)
        if print_stats:
            new_pop_name = action.new_population_name or f"{action.method}_{left.population_name}_{right.population_name}"
            new_pop = self.filegroup.get_population(population_name=new_pop_name)
            print(f"------ {action.action_name} ------")
            parent_n = self.filegroup.get_population(left.parent).n
            print(f"Parent ({left.parent}) n: {parent_n}")
            print(f"Left pop ({left.population_name}) n: {left.n}; {left.n / parent_n * 100}%")
            print(f"Right pop ({right.population_name}) n: {right.n}; {right.n / parent_n * 100}%")
            print(f"New population n: {new_pop.n}; {new_pop.n / parent_n * 100}%")
            print("-----------------------------------")
        if add_to_strategy:
            self.actions.append(action)

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
                           populations: str or list):
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
                  create_plot_kwargs: dict or None = None,
                  **kwargs):
        """
        Plot a gate. Must provide the name of a Gate currently associated to this GatingStrategy.
        This will plot the parent population this gate acts on along with the geometries
        that define the child populations the gate generates.

        Parameters
        ----------
        gate: str or Gate or EllipseGate or ThresholdGate or PolygonGate
        create_plot_kwargs: dict
            Keyword arguments for CreatePlot object. See CytoPy.plotting.CreatePlot for details.
        kwargs:
            Keyword arguments for plot_gate call.
            See CytoPy.plotting.CreatePlot.plot_population_geom for details.

        Returns
        -------
        Matplotlib.Axes
        """
        create_plot_kwargs = create_plot_kwargs or {}
        assert isinstance(gate, str), "Provide the name of an existing Gate in this GatingStrategy"
        assert gate in self.list_gates(), \
            f"Gate {gate} not recognised. Have you applied it and added it to the strategy?"
        gate = self.get_gate(gate=gate)
        parent = self.filegroup.load_population_df(population=gate.parent,
                                                   transform=None,
                                                   label_downstream_affiliations=False)
        plotting = CreatePlot(**create_plot_kwargs)
        return plotting.plot_population_geoms(parent=parent,
                                              children=[self.filegroup.get_population(c.name)
                                                        for c in gate.children],
                                              **kwargs)

    def plot_backgate(self,
                      parent: str,
                      overlay: list,
                      x: str,
                      y: str or None = None,
                      create_plot_kwargs: dict or None = None,
                      **kwargs):
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
            Additional keyword arguments passed to CytoPy.flow.plotting.CreatePlot
        kwargs
            Additional keyword arguments passed to CytoPy.flow.plotting.CreatePlot.backgate

        Returns
        -------
        Matplotlib.Axes
        """
        assert parent in self.list_populations(), "Parent population does not exist"
        assert all([x in self.list_populations() for x in overlay]), "One or more given populations could not be found"
        downstream = self.filegroup.list_downstream_populations(population=parent)
        assert all([x in downstream for x in overlay]), \
            "One or more of the given populations is not downstream of the given parent"
        plotting = CreatePlot(**create_plot_kwargs)
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
                                 **kwargs)

    def plot_population(self,
                        population: str,
                        x: str,
                        y: str or None = None,
                        transform_x: str or None = "logicle",
                        transform_y: str or None = "logicle",
                        create_plot_kwargs: dict or None = None,
                        **kwargs):
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
            Additional keyword arguments passed to CytoPy.flow.plotting.CreatePlot
        kwargs
            Additional keyword arguments passed to CytoPy.flow.plotting.CreatePlot.plot

        Returns
        -------
        Matplotlib.Axes
        """
        assert population in self.list_populations(), f"{population} does not exist"
        data = self.filegroup.load_population_df(population=population,
                                                 transform=None,
                                                 label_downstream_affiliations=False)
        create_plot_kwargs = create_plot_kwargs or {}
        plotting = CreatePlot(transform_x=transform_x,
                              transform_y=transform_y,
                              **create_plot_kwargs)
        return plotting.plot(data=data, x=x, y=y, **kwargs)

    def print_population_tree(self, **kwargs):
        """
        Print the population tree to stdout.
        Wraps CytoPy.data.fcs.FileGroup.print_population_tree

        Parameters
        ----------
        kwargs
            See keyword arguments for CytoPy.data.fcs.FileGroup.print_population_tree

        Returns
        -------
        None
        """
        self.filegroup.print_population_tree(**kwargs)

    def edit_gate(self,
                  gate_name: str,
                  x_threshold: float or None = None,
                  y_threshold: float or None = None,
                  x_values: list or None = None,
                  y_values: list or None = None):
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
        x_values: list
            Required for Polygon geometries
        y_values
            Required for Polygon geometries
        Returns
        -------
        None
        """
        gate = self.get_gate(gate=gate_name)
        err = "Cannot edit a gate that has not been applied; gate children not present in population " \
              "tree."
        assert all([x in self.filegroup.tree.keys() for x in [c.name for c in gate.children]]), err
        transforms = [gate.transformations.get(x, None) for x in ["x", "y"]]
        transforms = {k: v for k, v in zip([gate.x, gate.y], transforms) if k is not None}
        parent = self.filegroup.load_population_df(population=gate.parent,
                                                   transform=transforms)
        for child in gate.children:
            pop = self.filegroup.get_population(population_name=child.name)
            if isinstance(pop.geom, ThresholdGeom):
                assert x_threshold is not None, "For threshold geometry, please provide x_threshold"
                if pop.geom.y_threshold is not None:
                    assert y_threshold is not None, "For 2D threshold geometry, please provide y_threshold"
                update_threshold(population=pop,
                                 parent_data=parent,
                                 x_threshold=x_threshold,
                                 y_threshold=y_threshold)
            elif isinstance(pop.geom, PolygonGeom):
                assert x_values is not None and y_values is not None, \
                    "For polygon gate please provide x_values and y_values"
                update_polygon(population=pop,
                               parent_data=parent,
                               x_values=x_threshold,
                               y_values=y_threshold)
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
                update_threshold(population=pop,
                                 parent_data=parent,
                                 x_threshold=pop.geom.x_threshold,
                                 y_threshold=pop.geom.y_threshold)
            elif isinstance(pop.geom, PolygonGeom):
                update_polygon(population=pop,
                               parent_data=parent,
                               x_values=pop.geom.x_values,
                               y_values=pop.geom.y_values)

    def _control_gate(self,
                      gate: Gate or ThresholdGate or PolygonGate or EllipseGate):
        """
        Internal method for applying a gate using control data. Will first attempt to fetch the parent
        population for the control data (see CytoPy.data.fcs.FileGroup.load_ctrl_population_df)
        and then will fit the gate to this data. The resulting gate will be applied statically to
        the parent population from the primary data.

        Parameters
        ----------
        gate: Gate or ThresholdGate or PolygonGate or EllipseGate

        Returns
        -------
        list
            List of Populations
        """
        assert gate.ctrl in self.filegroup.controls, f"FileGroup does not have data for {gate.ctrl}"
        ctrl_parent_data = self.filegroup.load_ctrl_population_df(ctrl=gate.ctrl,
                                                                  population=gate.parent,
                                                                  transform=None)
        # Fit control data
        populations = gate.fit_predict(data=ctrl_parent_data)
        updated_children = list()
        for p in populations:
            eq_child = [c for c in gate.children if c.name == p.population_name]
            assert len(eq_child) == 1, "Invalid gate. Estimated populations do not match children."
            eq_child = eq_child[0]
            eq_child.geom = p.geom
            updated_children.append(eq_child)
        gate.children = updated_children
        # Predict original data
        parent_data = self.filegroup.load_population_df(population=gate.parent,
                                                        transform=None,
                                                        label_downstream_affiliations=False)
        return gate.fit_predict(data=parent_data)

    def save(self, *args, **kwargs):
        """
        Save GatingStrategy and the populations generated for the associated
        FileGroup.

        Parameters
        ----------
        args:
            Positional arguments for mongoengine.document.save call
        kwargs:
            Keyword arguments for mongoengine.document.save call

        Returns
        -------
        None
        """
        for g in self.gates:
            g.save()
        super().save(*args, **kwargs)
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
        remove_associations (default=True)
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
            self.print("Deleting gates...")
            for g in self.gates:
                g.delete()
        if remove_associations:
            self.print("Deleting associated populations in FileGroups...")
            for f in progress_bar(FileGroup.objects(), verbose=self.verbose):
                if self.name in f.gating_strategy:
                    f.gating_strategy = [gs for gs in f.gating_strategy if gs != self.name]
                    f.delete_populations(populations=populations)
                    f.save()
        self.print(f"{self.name} successfully deleted.")
