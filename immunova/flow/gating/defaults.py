import numpy as np
import mongoengine


class ChildConstructError(Exception):
    pass


class ChildPopulationCollection:
    """
    Collection of child populations. This is the standard input handed to the Gate object prior to gating. It defines
    the expected output of the operation.

    Attributes:
        - gate_type: the gate type of the intended gate to generate this child population. Must be one of:
            'threshold', 'cluster', 'geom'.
        - populations: dictionary of populations belonging to collection
    Methods:
        - add_population: add a population to the collection
        - remove_population: remove a population from the collection
    """
    def __init__(self, gate_type=None, json_dict=None):
        """
        Constructor for child population collection
        :param gate_type: the gate type of the intended gate to generate this child population. Must be one of:
            'threshold', 'cluster', 'geom'.
        """
        self.populations = dict()
        if json_dict is not None:
            self.deserialise(json_dict)
        else:
            try:
                assert gate_type in ['threshold_1d', 'threshold_2d', 'cluster', 'geom', None]
                self.gate_type = gate_type
            except AssertionError:
                print('Invalid gate type, must be one of: threshold_1d, threshold_2d, cluster, geom')

    def serialise(self):
        serialised = dict(gate_type=self.gate_type, populations=list())
        for pop in self.populations.keys():
            serialised['populations'].append(self.populations[pop].properties)
        return serialised

    def deserialise(self, json_dict):
        self.gate_type = json_dict['gate_type']
        for pop in json_dict['populations']:
            name = pop.pop('name')
            self.add_population(name=name, **pop)

    class ChildPopulation:
        def __init__(self, gate_type, **kwargs):
            """
            Constructor for ChildPopulation
            :param name: name of the population
            :param gate_type: the gate type of the intended gate to generate this child population. Must be one of:
            'threshold', 'cluster', 'geom'.
            :param kwargs: arguments for population definition. Will differ depending on gate type:
            threshold:
                - definition: one of ['+', '-','++', '--', '-+', '-+']. Defines how the population is identified in respect
                to the gate's resulting threshold(s)
            cluster:
                - target: 2d coordinate of expected population medoid
                - weight: integer value for populations ranked priority (used in case of merged populations)
            geom:
                - definition: one of ['+', '-']. Defines how the population is identified in respect
                to the gate's resulting geom
            """
            self.gate_type = gate_type
            x = self.__validate_input(gate_type, **kwargs)
            if x is False:
                raise ChildConstructError('Aborting; invalid child population construct')
            self.index = np.array([])
            self.geom = None

        def update_index(self, idx: np.array, merge_options: str = 'merge') -> None:
            """
            Update the index values of this population
            :param idx: index values corresponding to events data
            :param merge_options: how to handle existing data; either overwrite or merge
            :return: None
            """
            if merge_options == 'overwrite':
                self.index = idx
            elif merge_options == 'merge':
                self.index = np.concatenate((self.index, idx))
            else:
                print('Invalid input for merge_options, must be one of: merge, overwrite')

        def update_geom(self, x: str, shape: str or None = None, y: str or None = None, **kwargs):
            """
            Update geom associated to this child population instance
            :param shape: type of shape generated, current valid inputs are: ellipse, rect, threshold, 2d_threshold
            :param x: name of X dimension
            :param y: name of Y dimension (optional)
            :param kwargs: other parameters that describe this geometric object
            :return: None
            """
            self.geom = self.Geom(shape, x, y, **kwargs)

        def __validate_input(self, gate_type, **kwargs) -> bool:
            """
            Internal method. Called on init to validate input for child population defintion. If valid, kwargs will
            population the properties attribute of this child population. If invalid an AssertationError will be generated.
            :param gate_type: the gate type of the intended gate to generate this child population. Must be one of:
            'threshold', 'cluster', 'geom'.
            :param kwargs: arguments for population definition
            :return: None
            """
            try:
                if gate_type == 'threshold_1d' or gate_type == 'geom':
                    if not kwargs.keys() == {'definition', 'name'}:
                        raise ChildConstructError(f'For a threshold_1d gate child population must'
                                                  f' be defined with keys `definition` and `name`; {kwargs.keys()} '
                                                  f'provided')
                    if not kwargs['definition'] in ['-', '+']:
                        raise ChildConstructError(f'For a threshold_1d `definition` must be one of [+, -]'
                                                  f' not {kwargs["definition"]}')
                if gate_type == 'threshold_2d':
                    if not kwargs.keys() == {'definition', 'name'}:
                        raise ChildConstructError(f'For a threshold_2d gate child population must'
                                                  f' be defined with keys `definition` and `name`; {kwargs.keys()} '
                                                  f'provided')
                    if type(kwargs['definition']) == list or \
                            type(kwargs['definition']) == mongoengine.base.datastructures.BaseList:
                        for x in kwargs['definition']:
                            if x not in ['++', '--', '-+', '+-']:
                                raise ChildConstructError(f'Invalid definition {x}, must be one of '
                                                          f'[++, --, -+, +-]')
                    else:
                        if not kwargs['definition'] in ['++', '--', '-+', '+-']:
                            raise ChildConstructError(f'Invalid definition {kwargs["definition"]}, must be '
                                                      f'one of [++, --, -+, +-]')
                if gate_type == 'cluster':
                    if not kwargs.keys() == {'target', 'weight', 'name'}:
                        ChildConstructError(f'For cluster gating child population must contain keys [target, weight, '
                                            f'name]; {kwargs.keys()} provided')
                    if len(kwargs['target']) != 2:
                        ChildConstructError('Invalid target provided for child population, must be a tuple or list '
                                            f'of length two, not {kwargs["target"]}')
                    if not all([isinstance(x, int) or isinstance(x, float) for x in kwargs['target']]):
                        ChildConstructError('Invalid data type for target provided for child population, must be list '
                                            f'or tuple of floats or integers not {kwargs["target"]}')
                    if not isinstance(kwargs['weight'], int):
                        ChildConstructError('Invalid data type provided for weight of child population, must be '
                                            f'of type Integer not {kwargs["weight"]}')
            except ChildConstructError as e:
                print(e)
                return False
            self.properties = kwargs
            return True

        class Geom(dict):
            """
            Default definition for geometric object defining a population
            """
            def __init__(self, shape: str or None, x: str, y: str or None, **kwargs):
                """
                Constructor class for geometric object
                :param shape: type of shape generated, current valid inputs are: ellipse, rect, threshold, 2d_threshold,
                or None
                :param x: name of X dimension
                :param y: name of Y dimension (optional)
                :param kwargs: other parameters that describe this geometric object
                """
                super().__init__()
                try:
                    assert shape in ['ellipse', 'rect', 'threshold', '2d_threshold', 'poly']
                    self.shape = shape
                except AssertionError:
                    print('Invalid shape, must be one of: ellipse, rect, threshold, 2d_threshold', 'poly')
                self.x = x
                self.y = y
                for k, v in kwargs.items():
                    self[k] = v

            def as_dict(self) -> dict:
                """
                Convert object to base class dictionary
                :return: dictionary of geometric object properties
                """
                self.update({'shape': self.shape, 'x': self.x, 'y': self.y})
                return self

    def add_population(self, name: str, **kwargs) -> None:
        """
        Add a new population to this collection
        :param name: name of the population
        :param kwargs: arguments for population definition. Will differ depending on gate type:
            threshold:
                - definition: one of ['+', '-','++', '--', '-+', '-+']. Defines how the population is identified in respect
                to the gate's resulting threshold(s)
            cluster:
                - target: 2d coordinate of expected population medoid
                - weight: integer value for populations ranked priority (used in case of merged populations)
            geom:
                - definition: one of ['+', '-']. Defines how the population is identified in respect
                to the gate's resulting geom
        :return: None
        """
        if name in self.populations.keys():
            print(f'Error: a population with name {name} has already been associated to this '
                  f'ChildPopulationCollection')
            return None
        if self.gate_type == 'threshold_1d' or self.gate_type == 'threshold_2d':
            try:
                current_definitions = list()
                for _, d in self.populations.items():
                    if type(d.properties['definition']) == list:
                        current_definitions = current_definitions + d.properties['definition']
                    else:
                        current_definitions.append(d.properties['definition'])
                if any([kwargs['definition'] == x for x in current_definitions]):
                    print(f"Error: definition {kwargs['definition']} has already been assigned to a population "
                          f"in this collection")
            except KeyError:
                print(f'Invalid input for child population construction for gate type {self.gate_type}')
        self.populations[name] = self.ChildPopulation(name=name, gate_type=self.gate_type, **kwargs)

    def remove_population(self, name: str) -> None:
        """
        Remove population from collection
        :param name: name of population to remove
        :return: None
        """
        if name not in self.populations.keys():
            print(f'Error: population {name} does not exist')
        self.populations.pop(name)

    def fetch_by_definition(self, definition):
        for name, d in self.populations.items():
            if type(d.properties['definition']) == list:
                if definition in d.properties['definition']:
                    return name
            else:
                if definition == d.properties['definition']:
                    return name
        return None

