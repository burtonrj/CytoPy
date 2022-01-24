import logging
from typing import Dict
from typing import Optional

import pandas as pd
from anndata import AnnData
from scanpy.preprocessing import neighbors
from scanpy.tools import leiden

logger = logging.getLogger(__name__)


class Leiden:
    def __init__(self, knn_params: Optional[Dict] = None, **clustering_params):
        self.knn_params = knn_params or {}
        self.params = clustering_params
        self.params["key_added"] = "leiden"
        self.knn_params["n_pcs"] = 0
        self.knn_params["use_rep"] = None

    @staticmethod
    def _build_adata(data: pd.DataFrame) -> AnnData:
        return AnnData(data.values, var=data.columns.tolist())

    def _compute_neighbourhood_graph(self, adata: AnnData):
        neighbors(adata=adata, copy=False, **self.knn_params)

    def fit_predict(self, data: pd.DataFrame):
        logger.info("Converting data to annotated dataframe")
        adata = self._build_adata(data=data)
        logger.info("Constructing neighbourhood graph")
        self._compute_neighbourhood_graph(adata=adata)
        logger.info("Performing leiden clustering")
        leiden(adata=adata, **self.params)
        return adata.obs["leiden"].values
