#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module defines the logicle (biexponential) transform for Matplotlib plots

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
import numpy as np
import polars as pl
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Locator
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.ticker import NullFormatter

from cytopy.flow.transform import LogicleTransformer


__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


class LogicleScale(mscale.ScaleBase):
    name = "logicle"

    def __init__(self, axis, w: float = 0.5, m: float = 4.5, a: float = 0.0, t: int = 262144):
        super().__init__(axis=axis)
        self._scaler = LogicleTransformer(w=w, m=m, t=t, a=a)

    def get_transform(self):
        return self.LogicleTransform(self._scaler)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(LogicleMajorLocator())
        axis.set_major_formatter(LogFormatterMathtext(10))
        axis.set_minor_locator(LogicleMinorLocator())
        axis.set_minor_formatter(NullFormatter())

    class LogicleTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, scaler: LogicleTransformer):
            mtransforms.Transform.__init__(self)
            self._scaler = scaler

        def transform_non_affine(self, data):
            data = pd.DataFrame(data, columns=["x"])
            data = self._scaler.scale(data=data, features=["x"])
            return data.values

        def inverted(self):
            return LogicleScale.InvertedLogicalTransform(scaler=self._scaler)

    class InvertedLogicalTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, scaler: LogicleTransformer):
            mtransforms.Transform.__init__(self)
            self._scaler = scaler

        def transform_non_affine(self, data):
            data = pd.DataFrame(data, columns=["x"])
            data = self._scaler.inverse_scale(data=data, features=["x"])
            return data.values

        def inverted(self):
            return LogicleScale.LogicleTransform(scaler=self._scaler)


class LogicleMajorLocator(Locator):
    """
    Lifted from Cytoflow and authored by bpteague #Todo improve attribution
    Determine the tick locations for logicle axes.
    Based on matplotlib.LogLocator
    """

    def set_params(self, **kwargs):
        """Empty"""
        pass

    def __call__(self):
        "Return the locations of the ticks"
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        "Every decade, including 0 and negative"

        vmin, vmax = self.view_limits(vmin, vmax)
        kwargs = self.axis._scale._scaler.kwargs

        max_decade = np.ceil(np.log10(vmax * 1.1))
        min_positive_decade = np.ceil(np.log10(kwargs["t"]) - kwargs["m"]) + 1

        if vmin < 0:
            max_negative_decade = np.floor(np.log10(-1.0 * vmin))
            major_ticks = [-1.0 * 10 ** x for x in np.arange(max_negative_decade, 1, -1)]
            major_ticks.append(0.0)
        else:
            major_ticks = [0.0] if vmin == 0.0 else []

        major_ticks.extend([10 ** x for x in np.arange(min_positive_decade, max_decade, 1)])

        return self.raise_if_exceeds(np.asarray(major_ticks))

    def view_limits(self, data_min, data_max):
        "Try to choose the view limits intelligently"

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


class LogicleMinorLocator(Locator):
    """
    Lifted from Cytoflow and authored by bpteague #Todo impove attribution
    Determine the tick locations for logicle axes.
    Based on matplotlib.LogLocator
    """

    def set_params(self):
        """Empty"""
        pass

    def __call__(self):
        "Return the locations of the ticks"
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def tick_values(self, vmin, vmax):
        "Every tenth decade, including 0 and negative"

        vmin, vmax = self.view_limits(vmin, vmax)
        kwargs = self.axis._scale._scaler.kwargs

        max_decade = np.ceil(np.log10(vmax * 1.1)) + 1
        min_positive_decade = np.ceil(np.log10(kwargs["t"]) - kwargs["m"]) + 1

        if vmin < 0:
            max_negative_decade = np.floor(np.log10(-1.0 * vmin)) + 1
            major_ticks = [-1.0 * 10 ** x for x in np.arange(max_negative_decade, 1, -1)]
            major_ticks.append(0.0)
        else:
            major_ticks = [0.0] if vmin == 0.0 else []

        major_ticks.extend([10 ** x for x in np.arange(min_positive_decade, max_decade, 1)])

        major_tick_pairs = [(major_ticks[x], major_ticks[x + 1]) for x in range(len(major_ticks) - 1)]
        minor_ticks_lol = [np.arange(x, y, max(np.abs([x, y]) / 10)) for x, y in major_tick_pairs]
        minor_ticks = [item for sublist in minor_ticks_lol for item in sublist]

        return minor_ticks

    def view_limits(self, data_min, data_max):
        "Try to choose the view limits intelligently"

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


mscale.register_scale(LogicleScale)
