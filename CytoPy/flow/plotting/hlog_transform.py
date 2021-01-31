from ..transform import HyperlogTransformer
from matplotlib.ticker import NullFormatter, LogFormatterMathtext
from matplotlib.ticker import Locator
from matplotlib import transforms as mtransforms
from matplotlib import scale as mscale
import pandas as pd
import numpy as np


class HyperlogScale(mscale.ScaleBase):
    name = "hyperlog"

    def __init__(self, axis, w: float = 0.5, m: float = 4.5, t: int = 262144):
        super().__init__(axis=axis)
        self._scaler = HyperlogTransformer(w=w, m=m, t=t)

    def get_transform(self):
        return self.HyperlogTransform(self._scaler)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(HlogMajorLocator())
        axis.set_major_formatter(LogFormatterMathtext(10))
        axis.set_minor_locator(HlogMinorLocator())
        axis.set_minor_formatter(NullFormatter())

    class HyperlogTransform(mtransforms.Transform):
        def __init__(self, scaler: HyperlogTransformer):
            mtransforms.Transform.__init__(self)
            self._scaler = scaler

        def transform_non_affine(self, data):
            data = pd.DataFrame({"x": data})
            data = self._scaler.scale(data=data, features=["x"])
            return data.x.values

        def inverted(self):
            return HyperlogScale.InvertedHyperlogTransform(scaler=self._scaler)

    class InvertedHyperlogTransform(mtransforms.Transform):
        def __init__(self, scaler: HyperlogTransformer):
            mtransforms.Transform.__init__(self)
            self._scaler = scaler

        def transform_non_affine(self, data):
            data = pd.DataFrame({"x": data})
            data = self._scaler.inverse(data=data, features=["x"])
            return data.x.values

        def inverted(self):
            return HyperlogScale.HyperlogTransform(scaler=self._scaler)


class HlogMajorLocator(Locator):
    """
     Lifted from Cytoflow and authored by bpteague #Todo improve attribution
    Determine the tick locations for hlog axes.
    Based on matplotlib.LogLocator
    """

    def set_params(self):
        """Empty"""
        pass

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        'Every decade, including 0 and negative'

        vmin, vmax = self.view_limits(vmin, vmax)
        max_decade = 10 ** np.ceil(np.log10(vmax))

        if vmin < 0:
            min_decade = -1.0 * 10 ** np.floor(np.log10(-1.0 * vmin))
            ticks = [-1.0 * 10 ** x for x in np.arange(np.log10(-1.0 * min_decade), 1, -1)]
            ticks.append(0.0)
            ticks.extend([10 ** x for x in np.arange(2, np.log10(max_decade), 1)])
        else:
            ticks = [0.0] if vmin == 0.0 else []
            ticks.extend([10 ** x for x in np.arange(1, np.log10(max_decade), 1)])

        return self.raise_if_exceeds(np.asarray(ticks))

    def view_limits(self, data_min, data_max):
        'Try to choose the view limits intelligently'

        if data_max < data_min:
            data_min, data_max = data_max, data_min

        # get the nearest tenth-decade that contains the data

        if data_max > 0:
            logs = np.ceil(np.log10(data_max))
            vmax = np.ceil(data_max / (10 ** (logs - 1))) * (10 ** (logs - 1))
        else:
            vmax = 100

        if data_min >= 0:
            vmin = 0
        else:
            logs = np.ceil(np.log10(-1.0 * data_min))
            vmin = np.floor(data_min / (10 ** (logs - 1))) * (10 ** (logs - 1))

        return mtransforms.nonsingular(vmin, vmax)


class HlogMinorLocator(Locator):
    """
     Lifted from Cytoflow and authored by bpteague #Todo improve attribution
    Determine the tick locations for logicle axes.
    Based on matplotlib.LogLocator
    """

    def set_params(self):
        """Empty"""
        pass

    def __call__(self):
        'Return the locations of the ticks'
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        'Every tenth decade, including 0 and negative'

        vmin, vmax = self.view_limits(vmin, vmax)

        if vmin < 0:
            lt = [np.arange(10 ** x, 10 ** (x - 1), -1.0 * (10 ** (x - 1)))
                  for x in np.arange(np.ceil(np.log10(-1.0 * vmin)), 1, -1)]

            # flatten and take the negative
            lt = [-1.0 * item for sublist in lt for item in sublist]

            # whoops! missed an endpoint
            lt.extend([-10.0])

            gt = [np.arange(10 ** x, 10 ** (x + 1), 10 ** x)
                  for x in np.arange(1, np.log10(vmax))]

            # flatten
            gt = [item for sublist in gt for item in sublist]

            ticks = lt
            ticks.extend(gt)
        else:
            vmin = max((vmin, 1))
            ticks = [np.arange(10 ** x, 10 ** (x + 1), 10 ** x)
                     for x in np.arange(np.log10(vmin), np.log10(vmax))]
            ticks = [item for sublist in ticks for item in sublist]

        return self.raise_if_exceeds(np.asarray(ticks))


mscale.register_scale(HyperlogScale)
