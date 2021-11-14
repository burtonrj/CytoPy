"""
https://academic.oup.com/bioinformatics/article/36/9/2778/5714737
"""
import logging
import time
from typing import List
from typing import Optional
from typing import Union

import hnswlib
import igraph as ig
import leidenalg
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


logger = logging.getLogger(__name__)


class PARC:
    def __init__(
        self,
        dist_std_local: int = 3,
        jac_std_global: str = "median",
        keep_all_local_dist: str = "auto",
        max_cluster_size: Optional[float] = 0.4,
        smallest_population: int = 10,
        jac_weighted_edges: bool = True,
        jac_weighted_edges_large: bool = True,
        jac_std_large: float = 0.3,
        knn: int = 30,
        n_iter_leiden: int = 5,
        random_seed: int = 42,
        n_jobs: int = -1,
        distance: str = "l2",
        time_smallpop: int = 15,
        partition_type: str = "ModularityVP",
        resolution_parameter: int = 1.0,
        knn_struct: Optional[hnswlib.Index] = None,
        neighbor_graph: Optional[csr_matrix] = None,
        hnsw_param_ef_construction: int = 150,
    ):
        if resolution_parameter != 1:
            partition_type = "RBVP"
        self.dist_std_local = dist_std_local
        self.jac_std_global = jac_std_global
        self.jac_std_large = jac_std_large
        self.jac_weighted_edges_large = jac_weighted_edges_large
        self.keep_all_local_dist = keep_all_local_dist
        self.max_cluster_size = max_cluster_size
        self.small_pop = smallest_population
        self.jac_weighted_edges = jac_weighted_edges
        self.knn = knn
        self.n_iter_leiden = n_iter_leiden
        self.random_seed = random_seed
        self.num_threads = n_jobs
        self.distance = distance
        self.time_smallpop = time_smallpop
        self.partition_type = partition_type
        self.resolution_parameter = resolution_parameter
        self.knn_struct = knn_struct
        self.neighbor_graph = neighbor_graph
        self.hnsw_param_ef_construction = hnsw_param_ef_construction

    def _local_pruning(self, neighbor_array: np.ndarray, distance_array: np.ndarray):
        logger.info(
            f"Commencing local pruning based on Euclidean distance metric at"
            f" {self.dist_std_local}, 's.dev above mean"
        )
        n_neighbors = neighbor_array.shape[1]
        row_list = []
        col_list = []
        weight_list = []
        distance_array = distance_array + 0.1
        rowi = 0
        discard_count = 0
        for row in neighbor_array:
            distlist = distance_array[rowi, :]
            to_keep = np.where(distlist < np.mean(distlist) + self.dist_std_local * np.std(distlist))[0]  # 0*std
            updated_nn_ind = row[np.ix_(to_keep)]
            updated_nn_weights = distlist[np.ix_(to_keep)]
            discard_count = discard_count + (n_neighbors - len(to_keep))

            for ik in range(len(updated_nn_ind)):
                if rowi != row[ik]:  # remove self-loops
                    row_list.append(rowi)
                    col_list.append(updated_nn_ind[ik])
                    dist = np.sqrt(updated_nn_weights[ik])
                    weight_list.append(1 / (dist + 0.1))
            rowi = rowi + 1
        return row_list, col_list, weight_list

    def _build_csrmatrix(self, neighbor_array: np.ndarray, distance_array: np.ndarray):
        # neighbor array not listed in in any order of proximity
        n_cells = neighbor_array.shape[0]
        if not self.keep_all_local_dist:  # locally prune based on (squared) l2 distance
            row_list, col_list, weight_list = self._local_pruning(neighbor_array, distance_array)
        else:  # dont prune based on distance
            row_list = []
            n_neighbors = neighbor_array.shape[1]
            row_list.extend(list(np.transpose(np.ones((n_neighbors, n_cells)) * range(0, n_cells)).flatten()))
            col_list = neighbor_array.flatten().tolist()
            weight_list = (1.0 / (distance_array.flatten() + 0.1)).tolist()

        csr_graph = csr_matrix(
            (np.array(weight_list), (np.array(row_list), np.array(col_list))), shape=(n_cells, n_cells)
        )
        return csr_graph

    def _construct_knn(self, data: np.ndarray):
        ef_query = max(100, self.knn + 1)  # ef always should be >K. higher ef, more accurate query
        n_elements, num_dims = data.shape
        p = hnswlib.Index(space=self.distance, dim=num_dims)
        p.set_num_threads(self.num_threads)
        if n_elements < 10000:
            ef_query = min(n_elements - 10, 500)
            ef_construction = ef_query
        else:
            ef_construction = self.hnsw_param_ef_construction
        if (num_dims > 30) & (n_elements <= 50000):
            p.init_index(
                max_elements=n_elements, ef_construction=ef_construction, M=48
            )  # good for scRNA seq where dimensionality is high
        else:
            p.init_index(max_elements=n_elements, ef_construction=ef_construction, M=24)  # 30
        p.add_items(data)
        p.set_ef(ef_query)  # ef should always be > k
        return p

    def _construct_knn_big(self, data: np.ndarray):
        ef_query = max(100, self.knn + 1)  # ef always should be >K. higher ef, more accurate query
        n_elements, num_dims = data.shape
        p = hnswlib.Index(space="l2", dim=num_dims)
        p.set_num_threads(self.num_threads)
        p.init_index(max_elements=n_elements, ef_construction=200, M=30)
        p.add_items(data)
        p.set_ef(ef_query)  # ef should always be > k
        return p

    def _global_pruning(self, n: int, sim_list: List, edgelist: List):
        sim_list_array = np.asarray(sim_list)
        edge_list_array = np.asarray(edgelist)

        if self.jac_std_global == "median":
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - self.jac_std_global * np.std(sim_list)
        strong_locs = np.where(sim_list_array > threshold)[0]
        new_edgelist = list(edge_list_array[strong_locs])
        sim_list_new = list(sim_list_array[strong_locs])
        g_sim = ig.Graph(n=n, edges=list(new_edgelist), edge_attrs={"weight": sim_list_new})
        g_sim.simplify(combine_edges="sum")
        return g_sim

    def _community_detection(self, graph: ig.Graph, n_elements: int):
        kwargs = {}
        if self.jac_weighted_edges:
            kwargs["weights"] = "weight"
        if self.partition_type == "ModularityVP":
            logger.info("Using MVP partition type")
            partition_type = leidenalg.ModularityVertexPartition
        else:
            logger.info("Using RBC partition type")
            partition_type = leidenalg.RBConfigurationVertexPartition
            kwargs["resolution_parameter"] = self.resolution_parameter

        partition = leidenalg.find_partition(
            graph, partition_type, n_iterations=self.n_iter_leiden, seed=self.random_seed, **kwargs
        )
        parc_labels_leiden = np.asarray(partition.membership)
        parc_labels_leiden = np.reshape(parc_labels_leiden, (n_elements, 1))
        return parc_labels_leiden

    def _run_parc_large_cluster(self, data: np.ndarray):
        hnsw = self._construct_knn_big(data=data)
        knn = self.knn if data.shape[0] > self.knn else int(max(5, 0.2 * data.shape[0]))
        neighbor_array, distance_array = hnsw.knn_query(data, k=knn)
        csr_array = self._build_csrmatrix(neighbor_array, distance_array)
        sources, targets = csr_array.nonzero()
        mask = np.zeros(len(sources), dtype=bool)
        # smaller distance means stronger edge
        mask |= csr_array.data > (np.mean(csr_array.data) + np.std(csr_array.data) * 5)
        csr_array.data[mask] = 0
        csr_array.eliminate_zeros()
        sources, targets = csr_array.nonzero()
        edgelist = list(zip(sources.tolist(), targets.tolist()))
        g_sim = ig.Graph(edgelist, edge_attrs={"weight": csr_array.data.tolist()})
        sim_list = g_sim.similarity_jaccard(pairs=edgelist.copy())  # list of jaccard weights

        new_edgelist = []
        sim_list_array = np.asarray(sim_list)
        if self.jac_std_large == "median":
            threshold = np.median(sim_list)
        else:
            threshold = np.mean(sim_list) - self.jac_std_large * np.std(sim_list)
        logger.info("jac threshold %.3f" % threshold)
        logger.info("jac std %.3f" % np.std(sim_list))
        logger.info("jac mean %.3f" % np.mean(sim_list))
        strong_locs = np.where(sim_list_array > threshold)[0]
        for ii in strong_locs:
            new_edgelist.append(edgelist[ii])
        sim_list_new = list(sim_list_array[strong_locs])
        if self.jac_weighted_edges_large:
            g_sim = ig.Graph(n=data.shape[0], edges=list(new_edgelist), edge_attrs={"weight": sim_list_new})
        else:
            g_sim = ig.Graph(n=data.shape[0], edges=list(new_edgelist))
        g_sim.simplify(combine_edges="sum")
        parc_labels_leiden = self._community_detection(graph=g_sim, n_elements=data.shape[0])
        parc_labels_leiden = self._review_small_clusters(
            parc_labels_leiden=parc_labels_leiden, neighbor_array=neighbor_array
        )
        _, parc_labels_leiden = np.unique(list(parc_labels_leiden.flatten()), return_inverse=True)
        return parc_labels_leiden

    def _fragment_large_clusters(
        self,
        data: np.ndarray,
        cluster_big_loc: np.ndarray,
        parc_labels_leiden: np.ndarray,
    ):
        list_pop_too_big = [len(np.where(parc_labels_leiden == 0)[0])]
        too_big = True
        while too_big:
            parc_labels_leiden_big = self._run_parc_large_cluster(data=data[cluster_big_loc, :])
            parc_labels_leiden_big = parc_labels_leiden_big + 100000
            pop_list = []
            for item in set(list(parc_labels_leiden_big.flatten())):
                pop_list.append([item, list(parc_labels_leiden_big.flatten()).count(item)])
            logger.info(f"Population of big clusters: {pop_list}")
            jj = 0
            logger.info(f"Shape PARC_labels_leiden: {parc_labels_leiden.shape}")
            for j in cluster_big_loc:
                parc_labels_leiden[j] = parc_labels_leiden_big[jj]
                jj = jj + 1
            dummy, parc_labels_leiden = np.unique(list(parc_labels_leiden.flatten()), return_inverse=True)
            logger.info(f"New set of labels {np.unique(parc_labels_leiden.flatten())}")
            too_big = False
            set_parc_labels_leiden = np.unique(parc_labels_leiden.flatten())

            parc_labels_leiden = np.asarray(parc_labels_leiden)
            for cluster_ii in set_parc_labels_leiden:
                cluster_ii_loc = np.where(parc_labels_leiden == cluster_ii)[0]
                pop_ii = len(cluster_ii_loc)
                not_yet_expanded = pop_ii not in list_pop_too_big
                if pop_ii > self.max_cluster_size * data.shape[0] and not_yet_expanded:
                    too_big = True
                    logger.info(f"Cluster {cluster_ii} is too big with population size {pop_ii}, it will be expanded.")
                    cluster_big_loc = cluster_ii_loc
                    big_pop = pop_ii
                    list_pop_too_big.append(big_pop)
        _, parc_labels_leiden = np.unique(list(parc_labels_leiden.flatten()), return_inverse=True)
        return parc_labels_leiden

    def _collect_small_clusters(self, parc_labels_leiden: np.ndarray):
        small_pop_list = []
        small_cluster_list = []
        small_pop_exist = False
        for cluster in np.unique(parc_labels_leiden.flatten()):
            population = len(np.where(parc_labels_leiden == cluster)[0])
            if population < self.small_pop:  # 10
                small_pop_exist = True
                small_pop_list.append(list(np.where(parc_labels_leiden == cluster)[0]))
                small_cluster_list.append(cluster)
        return small_pop_list, small_cluster_list, small_pop_exist

    @staticmethod
    def _match_small_clusters_to_neighbourhoods(
        small_pop_list: List, small_cluster_list: List, neighbor_array: np.ndarray, parc_labels_leiden: np.ndarray
    ):
        for small_cluster in small_pop_list:
            for single_cell in small_cluster:
                old_neighbors = neighbor_array[single_cell]
                group_of_old_neighbors = parc_labels_leiden[old_neighbors]
                group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                available_neighbours = set(group_of_old_neighbors) - set(small_cluster_list)
                if len(available_neighbours) > 0:
                    available_neighbours_list = [
                        value for value in group_of_old_neighbors if value in list(available_neighbours)
                    ]
                    best_group = max(available_neighbours_list, key=available_neighbours_list.count)
                    parc_labels_leiden[single_cell] = best_group

    def _review_small_clusters(self, parc_labels_leiden: np.ndarray, neighbor_array: np.ndarray):
        small_pop_list, small_cluster_list, small_pop_exist = self._collect_small_clusters(
            parc_labels_leiden=parc_labels_leiden
        )
        self._match_small_clusters_to_neighbourhoods(
            small_pop_list=small_pop_list,
            small_cluster_list=small_cluster_list,
            neighbor_array=neighbor_array,
            parc_labels_leiden=parc_labels_leiden,
        )
        time_smallpop_start = time.time()
        while small_pop_exist & ((time.time() - time_smallpop_start) < self.time_smallpop):
            small_pop_list = []
            small_pop_exist = False
            for cluster in set(list(parc_labels_leiden.flatten())):
                population = len(np.where(parc_labels_leiden == cluster)[0])
                if population < self.small_pop:
                    small_pop_exist = True
                    logger.info(f"Cluster {cluster} is below minimum population size limit of n={self.small_pop}")
                    small_pop_list.append(np.where(parc_labels_leiden == cluster)[0])
            for small_cluster in small_pop_list:
                for single_cell in small_cluster:
                    old_neighbors = neighbor_array[single_cell]
                    group_of_old_neighbors = parc_labels_leiden[old_neighbors]
                    group_of_old_neighbors = list(group_of_old_neighbors.flatten())
                    best_group = max(set(group_of_old_neighbors), key=group_of_old_neighbors.count)
                    parc_labels_leiden[single_cell] = best_group
        return parc_labels_leiden

    def _run(self, data: np.ndarray):
        logger.info("Building KNN structure")
        if self.neighbor_graph is not None:
            csr_array = self.neighbor_graph
            neighbor_array = np.split(self.neighbor_graph.indices, self.neighbor_graph.indptr)[1:-1]
        else:
            if self.knn > 190:
                logger.warning("Consider using a lower k for KNN graph construction for faster computation")
            knn_struct = self._construct_knn(data=data)
            neighbor_array, distance_array = knn_struct.knn_query(data, k=self.knn)
            csr_array = self._build_csrmatrix(neighbor_array, distance_array)

        logger.info("Constructing graph")
        sources, targets = csr_array.nonzero()
        edgelist = list(zip(sources, targets))
        graph = ig.Graph(edgelist, edge_attrs={"weight": csr_array.data.tolist()})

        logger.info("Commencing global pruning")
        g_sim = self._global_pruning(
            n=data.shape[0], sim_list=graph.similarity_jaccard(pairs=edgelist.copy()), edgelist=edgelist.copy()
        )

        logger.info("Commencing community detection")
        parc_labels_leiden = self._community_detection(graph=g_sim, n_elements=data.shape[0])

        if self.max_cluster_size is not None:
            largest_pop_n = len(np.where(parc_labels_leiden == 0)[0])
            if largest_pop_n > (self.max_cluster_size * data.shape[0]):
                logger.info(
                    f"0th cluster exceeds specified limit of {self.max_cluster_size*100}% of n;"
                    f"{largest_pop_n} > {self.max_cluster_size * data.shape[0]}"
                )
                logger.info("KNN will be reconstructed and large clusters fragmented to improve resolution")
                parc_labels_leiden = self._fragment_large_clusters()

        logger.info("Reviewing singletons and small clusters for outlier discrimination")
        parc_labels_leiden = self._review_small_clusters(
            parc_labels_leiden=parc_labels_leiden, neighbor_array=neighbor_array
        )
        _, parc_labels_leiden = np.unique(list(parc_labels_leiden.flatten()), return_inverse=True)
        parc_labels_leiden = list(parc_labels_leiden.flatten())
        pop_list = []
        for item in set(parc_labels_leiden):
            pop_list.append((item, parc_labels_leiden.count(item)))
        logger.info(f"Identified {len(pop_list)} clusters: {pop_list}")
        return parc_labels_leiden

    def fit_predict(self, data: Union[pd.DataFrame, np.ndarray]):
        logger.info(f"Performing PARC analysis; n={data.shape[0]}, d={data.shape[1]}")
        if isinstance(data, pd.DataFrame):
            data = data.values
        if self.keep_all_local_dist == "auto":
            if data.shape[0] > 300000:
                self.keep_all_local_dist = True
                logger.info("Data is larger than 300k, skipping local pruning to optimise performance.")
            else:
                self.keep_all_local_dist = False
        labels = self._run(data=data)
        logger.info(f"jac_std_global={self.jac_std_global}; dist_std_local={self.dist_std_local}")
        logger.info("Analysis complete!")
        return labels
