import logging
from typing import Dict
from typing import Optional
from typing import Union

import pandas as pd
from hdbscan import HDBSCAN
from sklearn.base import ClusterMixin
from sklearn.cluster import KMeans

from cytopy.clustering.consensus_k import KConsensusClustering
from cytopy.utils import DimensionReduction
from cytopy.utils.sampling import density_dependent_downsampling
from cytopy.utils.sampling import uniform_downsampling
from cytopy.utils.sampling import upsample_knn

logger = logging.getLogger(__name__)


class LatentClustering:
    def __init__(
        self,
        dim_reduction_method: str = "PHATE",
        dim_reduction_kwargs: Optional[Dict] = None,
        sample_size: int = 100000,
        density_dependent_downsample: bool = False,
        density_sampling_kwargs: Optional[Dict] = None,
        upsampling_kwargs: Optional[Dict] = None,
        cluster_model: Optional[ClusterMixin] = None,
        consensus_clustering: bool = False,
        consensus_clustering_params: Optional[Dict] = None,
    ):
        self.reducer_method = dim_reduction_method
        self.reducer_kwargs = dim_reduction_kwargs or {}
        self.sample_size = sample_size
        self.density_dependent_downsample = density_dependent_downsample
        self.density_sampling_kwargs = density_sampling_kwargs or {}
        self.upsampling_kwargs = upsampling_kwargs or {}

        if consensus_clustering:
            if cluster_model is None:
                cluster_model = KMeans(random_state=42)
            consensus_clustering_params = consensus_clustering_params or {
                "smallest_cluster_n": 5,
                "largest_cluster_n": 50,
            }
            self.model = KConsensusClustering(cluster=cluster_model, **consensus_clustering_params)
        else:
            self.model = cluster_model
            if cluster_model is None:
                self.model = HDBSCAN(min_cluster_size=15)

    def fit_predict(self, data: pd.DataFrame):
        if self.density_dependent_downsample:
            logger.info("Performing density dependent down-sampling")
            sample = density_dependent_downsampling(
                data=data, sample_size=self.sample_size, **self.density_sampling_kwargs
            )
        else:
            logger.info("Performing uniform down-sampling")
            sample = uniform_downsampling(data=data, sample_size=self.sample_size)
        logger.info("Fitting dimension reduction model")
        reducer = DimensionReduction(method=self.reducer_method, **self.reducer_kwargs)
        sample = reducer.fit_transform(data=sample, features=sample.columns)
        features = [x for x in sample.columns if self.reducer_method in x]
        logger.info(f"Clustering in down-sampled embedded space ({features})")
        labels = self.model.fit_predict(sample[features].values)
        logger.info("Up-sampling to original space")
        labels = upsample_knn(
            sample=sample,
            original_data=data,
            labels=labels,
            features=data.columns,
            **self.upsampling_kwargs,
        )
        return labels
