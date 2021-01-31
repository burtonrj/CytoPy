from ..transform import LogTransformer
from matplotlib.ticker import NullFormatter, LogFormatterMathtext, LogLocator
from matplotlib import transforms as mtransforms
from matplotlib import scale as mscale
import pandas as pd


class LogScale(mscale.ScaleBase):
    name = "log_scale"

    def __init__(self, axis, base="parametrized", m: float = 4.5, t: int = 262144, **kwargs):
        self._base = base
        super().__init__(axis=axis)
        self._formatting_kwargs = kwargs or {}
        if base == "parametrized":
            self._base = 10
        if base == "natural":
            self._base = 2.718
        self._scaler = LogTransformer(base=base, m=m, t=t)

    def get_transform(self):
        return self.LogTransform(scaler=self._scaler)

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(LogLocator(base=self._base, **self._formatting_kwargs))
        axis.set_major_formatter(LogFormatterMathtext(10))
        axis.set_minor_locator(LogLocator(base=self._base, **self._formatting_kwargs))
        axis.set_minor_formatter(NullFormatter())

    class LogTransform(mtransforms.Transform):
        def __init__(self, scaler: LogTransformer):
            mtransforms.Transform.__init__(self)
            self._scaler = scaler

        def transform_non_affine(self, data):
            data = pd.DataFrame({"x": data})
            data = self._scaler.scale(data=data, features=["x"])
            return data.x.values

        def inverted(self):
            return LogScale.InvertedLogTransform(scaler=self._scaler)

    class InvertedLogTransform(mtransforms.Transform):
        def __init__(self, scaler: LogTransformer):
            mtransforms.Transform.__init__(self)
            self._scaler = scaler

        def transform_non_affine(self, data):
            data = pd.DataFrame({"x": data})
            data = self._scaler.inverse(data=data, features=["x"])
            return data.x.values

        def inverted(self):
            return LogScale.LogTransform(scaler=self._scaler)


mscale.register_scale(LogScale)
