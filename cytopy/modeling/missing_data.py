from typing import Optional

import miceforest as mf
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler


def _encode_categorical(data: pd.DataFrame):
    for c in pd.concat([data.select_dtypes("object"), data.select_dtypes("bool")], axis=1):
        data[c] = data[c].astype("category")
        data[c] = data[c].cat.codes.apply(lambda x: None if x == -1 else x)
    return data


class ImputationBenchmarking:
    def __init__(
        self,
        complete_case: pd.DataFrame,
        dv: str,
        perc: float = 0.25,
        random_state: int = 42,
        scale: Optional[str] = "standard",
    ):
        complete_case = _encode_categorical(data=complete_case)
        self.features = [x for x in complete_case.columns if x != dv]
        if scale == "standard":
            complete_case[self.features] = StandardScaler().fit_transform(complete_case[self.features])
        elif scale == "norm":
            complete_case[self.features] = MinMaxScaler().fit_transform(complete_case[self.features])
        elif scale == "robust":
            complete_case[self.features] = RobustScaler().fit_transform(complete_case[self.features])

        self.complete_case = complete_case.copy()
        self.amputated = mf.ampute_data(
            data=complete_case, variables=self.features, perc=perc, random_state=random_state
        )

    def complete_case_performance(self, estimator: BaseEstimator):
        pass

    def benchmark_mice(self, estimator: BaseEstimator):
        pass

    def benchmark_miceforest(self):
        pass
