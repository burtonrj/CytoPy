import numpy as np


class Geom(dict):
    def __init__(self, shape: str or None, x: str, y: str or None, **kwargs):
        super().__init__()
        self.shape = shape
        self.x = x
        self.y = y
        for k, v in kwargs.items():
            self[k] = v

    def to_mongo(self):
        output = [(k, v) for k, v in self.items()]
        output.append(('shape', self.shape))
        output.append(('x', self.x))
        output.append(('y', self.y))
        return output


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





