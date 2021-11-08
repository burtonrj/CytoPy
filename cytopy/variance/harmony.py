import logging
from typing import Iterable
from typing import List
from typing import Optional
from typing import Union

import harmonypy
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cytopy.utils.transform import Scaler
from cytopy.utils.transform import Transformer
from cytopy.variance.base import BatchCorrector

logger = logging.getLogger(__name__)


def batch_lisi(
    data: pd.DataFrame,
    meta: pd.DataFrame,
    features: List[str],
    meta_var: str = "sample_id",
    sample: Optional[int] = 1000,
    idx: Optional[Iterable[int]] = None,
):
    if meta_var not in meta.columns:
        logger.error(f"{meta_var} missing from meta attribute")
        raise ValueError(f"{meta_var} missing from meta attribute")
    data["meta"] = meta[meta_var]

    if idx is not None:
        data = data.iloc[idx]
    elif sample is not None:
        data = data.sample(n=sample)

    return harmonypy.lisi.compute_lisi(
        data[features].values,
        metadata=data[["meta"]],
        label_colnames=["meta"],
    )


class Harmony(BatchCorrector):
    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        transformer: Optional[Transformer] = None,
        scaler: Optional[Scaler] = None,
        verbose: bool = True,
    ):
        super().__init__(data, features, transformer, scaler, verbose)
        self.meta = self.original_data[["sample_id"]]
        self.harmony = None

    @property
    def corrected_data(self):
        if self.harmony is None:
            raise ValueError("Call 'run' to generate corrected data.")
        corrected = pd.DataFrame(self.harmony.Z_corr.T, columns=self.features)
        corrected["sample_id"] = self.meta.sample_id
        return corrected

    def run(self, var_use: str = "sample_id", **kwargs):
        """
        Run the harmony algorithm (see https://github.com/slowkow/harmonypy for details). Resulting object
        is stored in 'harmony' attribute

        Parameters
        ----------
        var_use: str (default="sample_id")
            Name of the meta variable to use to match batches
        kwargs:
            Additional keyword arguments passed to harmonypy.run_harmony

        Returns
        -------
        Harmony
        """
        logger.info("Running harmony")
        data = self.original_data[self.features]
        self.harmony = harmonypy.run_harmony(data_mat=data.to_numpy(), meta_data=self.meta, vars_use=var_use, **kwargs)
        return self

    def add_meta_variable(self, key: Union[str, List[str]], var_name: str):
        super(Harmony, self).add_meta_variable(key=key, var_name=var_name)
        self.meta[var_name] = self.original_data[var_name]

    def batch_lisi_distribution(self, meta_var: str = "sample_id", sample: Optional[int] = 1000, **kwargs):
        """
        Plot the distribution of LISI using the given meta_var as label

        Parameters
        ----------
        meta_var: str (default="sample_id")
        sample: float (default=1.)
            Number of events to sample prior to calculating LISI. Downsampling is recommended
            for large datasets, since LISI is computationally expensive
        kwargs:
            Additional keyword arguments passed to Seaborn.histplot

        Returns
        -------
        Matplotlib.Axes
        """
        data = self.original_data[self.features].copy()
        data["meta"] = self.meta[meta_var]
        idx = None
        if sample is not None:
            data = data.sample(n=sample)
            idx = data.index.values
        before = harmonypy.lisi.compute_lisi(
            data[self.features].values,
            metadata=data[["meta"]],
            label_colnames=["meta"],
        )
        data = pd.DataFrame(
            {
                "Before": before.reshape(-1),
                "After": batch_lisi(
                    data=self.corrected_data, meta=self.meta, features=self.features, meta_var=meta_var, idx=idx
                ).reshape(-1),
            }
        )
        data = data.melt(var_name="Data", value_name="LISI")
        kwargs = kwargs or {}
        kwargs["ax"] = kwargs.get("ax", plt.subplots(figsize=(5, 5))[1])
        return sns.histplot(data=data, x="LISI", hue="Data", **kwargs)
