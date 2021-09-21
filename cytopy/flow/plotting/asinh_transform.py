#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module defines the hyperbolic arcsine transform for Matplotlib plots

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
import pandas as pd
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.ticker import NullFormatter

from .hlog_transform import HlogMajorLocator
from .hlog_transform import HlogMinorLocator
from cytopy.flow.transform import AsinhTransformer


class AsinhScale(mscale.ScaleBase):
    name = "asinh"

    def __init__(self, axis, t: int = 262144, m: float = 4.5, a: float = 0, **kwargs):
        super().__init__(axis=axis)
        self._formatting_kwargs = kwargs or {}
        self._scaler = AsinhTransformer(m=m, t=t, a=a)

    def get_transform(self):
        return self.AsinhTransform(scaler=self._scaler)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(HlogMajorLocator())
        axis.set_major_formatter(LogFormatterMathtext(10))
        axis.set_minor_locator(HlogMinorLocator())
        axis.set_minor_formatter(NullFormatter())
        pass

    class AsinhTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, scaler: AsinhTransformer):
            mtransforms.Transform.__init__(self)
            self._scaler = scaler

        def transform_non_affine(self, data):
            data = pd.DataFrame(data, columns=["x"])
            data = self._scaler.scale(data=data, features=["x"])
            return data.values

        def inverted(self):
            return AsinhScale.InvertedAsinhTransform(scaler=self._scaler)

    class InvertedAsinhTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, scaler: AsinhTransformer):
            mtransforms.Transform.__init__(self)
            self._scaler = scaler

        def transform_non_affine(self, data):
            data = pd.DataFrame(data, columns=["x"])
            data = self._scaler.inverse_scale(data=data, features=["x"])
            return data.values

        def inverted(self):
            return AsinhScale.AsinhTransform(scaler=self._scaler)


mscale.register_scale(AsinhScale)
