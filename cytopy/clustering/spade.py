import logging
from typing import Dict
from typing import Optional
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import ClusterMixin
from sklearn.cluster import AgglomerativeClustering

from cytopy.clustering.consensus_k import KConsensusClustering
from cytopy.utils.sampling import density_dependent_downsampling
from cytopy.utils.sampling import upsample_knn

logger = logging.getLogger(__name__)


class CytoSPADE:
    def __init__(
        self,
        min_k: int = 10,
        max_k: int = 20,
        sample_size: int = 10000,
        sampling_alpha: int = 5,
        sampling_distance_metric: str = "manhattan",
        sampling_tree_size: int = 1000,
        outlier_dens: int = 1,
        target_dens: int = 5,
        density_dependent_sampling: bool = True,
        clustering_method: Optional[Type] = None,
        consensus_clustering: bool = True,
        cluster_params: Optional[Dict] = None,
        consensus_params: Optional[Dict] = None,
        upsampling_kwargs: Optional[Dict] = None,
    ):
        cluster_params = cluster_params or {}
        consensus_params = consensus_params or {}
        if clustering_method is None:
            _model = AgglomerativeClustering(**cluster_params)
        else:
            _model = clustering_method(**cluster_params)
        if consensus_clustering:
            self.model = KConsensusClustering(
                cluster=_model, smallest_cluster_n=min_k, largest_cluster_n=max_k, **consensus_params
            )
        else:
            self.model = _model
        self.consensus_clustering = consensus_clustering
        self.sample_size = sample_size
        self.sampling_alpha = sampling_alpha
        self.sampling_distance_metric = sampling_distance_metric
        self.sampling_tree_size = sampling_tree_size
        self.outlier_dens = outlier_dens
        self.target_dens = target_dens
        self.upsampling_kwargs = upsampling_kwargs or {}
        self.density_dependent_sampling = density_dependent_sampling

    def fit_predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        if self.density_dependent_sampling:
            logger.info(f"Density dependent down-sampling of input data to {self.sample_size} events")
            sample = density_dependent_downsampling(
                data=data,
                sample_size=self.sample_size,
                alpha=self.sampling_alpha,
                distance_metric=self.sampling_distance_metric,
                tree_sample=self.sampling_tree_size,
                outlier_dens=self.outlier_dens,
                target_dens=self.target_dens,
            )
        else:
            logger.info(f"Uniform down-sampling of input data to {self.sample_size} events")
            if data.shape[0] <= self.sample_size:
                raise ValueError(f"Cannot sample {self.sample_size} events from array with {data.shape[0]} rows.")
            sample = pd.DataFrame(data).sample(n=self.sample_size)
        logger.info(f"Clustering data")
        labels = self.model.fit_predict(sample)
        logger.info("Up-sampling clusters using KNN")
        labels = upsample_knn(
            sample=sample, original_data=data, labels=labels, features=data.columns, **self.upsampling_kwargs
        )
        logger.info("Clustering complete!")
        return labels
