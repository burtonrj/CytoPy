from ..transform import AsinhTransformer
from .hlog_transform import HlogMajorLocator, HlogMinorLocator
from matplotlib.ticker import NullFormatter, LogFormatterMathtext
from matplotlib import transforms as mtransforms
from matplotlib import scale as mscale
import pandas as pd


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
