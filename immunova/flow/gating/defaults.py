import numpy as np


class Geom(dict):
    def __init__(self, shape: str or None, x: str, y: str or None, **kwargs):
        super().__init__()
        self.shape = shape
        self.x = x
        self.y = y
        for k, v in kwargs.items():
            self[k] = v

    def as_dict(self):
        self.update({'shape': self.shape, 'x': self.x, 'y': self.y})
        return self


class GateOutput:
    def __init__(self):
        self.child_populations = dict()
        self.warnings = list()
        self.error = False
        self.error_msg = None

    def log_error(self, msg):
        self.error = True
        self.error_msg = msg

    def add_child(self, name: str, idx: np.array, geom: Geom or None = None, merge_options='merge'):
        if name in self.child_populations.keys():
            if merge_options == 'overwrite':
                self.child_populations.pop(name)
            else:
                self.child_populations['name']['index'] = np.concatenate(self.child_populations['name']['index'], idx)
                return None
        self.child_populations[name] = dict(index=idx, geom=geom)

#class ChildPopulationCollection:
#    def __init__(self):

class ChildPopulation:
    def __init__(self, name: str, gate_type, **kwargs):
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
        self.name = name
        self.gate_type = gate_type
        self.__validate_input(gate_type, **kwargs)

    def __validate_input(self, gate_type, **kwargs) -> None:
        """
        Internal method. Called on init to validate input for child population defintion. If valid, kwargs will
        population the properties attribute of this child population. If invalid an AssertationError will be generated.
        :param gate_type: the gate type of the intended gate to generate this child population. Must be one of:
        'threshold', 'cluster', 'geom'.
        :param kwargs: arguments for population definition
        :return: None
        """
        try:
            def check_keys(keys):
                for _, x_ in kwargs.items():
                    assert x_.keys() == set(keys)
            if gate_type == 'threshold':
                check_keys(['definition'])
                assert kwargs['definition'] in ['++', '--', '-+', '-+', '-', '+']
            if gate_type == 'cluster':
                check_keys(['target', 'weight'])
                assert len(kwargs['target']) == 2
                assert len(kwargs['weight']) == 1
                assert all([isinstance(x, int) or isinstance(x, float) for x in kwargs['target']])
                assert all([isinstance(x, int) or isinstance(x, float) for x in kwargs['weight']])
            if gate_type == 'geom':
                check_keys(['definition'])
                assert kwargs['definition'] in ['-', '+']
            self.properties = {k: v for k, v in kwargs}
        except AssertionError:
            print(f'Invalid input for child population construction for gate type {gate_type}')




