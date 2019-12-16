import mongoengine


class ClusteringDefinition(mongoengine.Document):
    """
    Defines the methodology and parameters of clustering to apply to an FCS File Group, or in the case of
    meta-clustering, a collection of FCS File Groups from the same FCS Experiment

    Parameters
    clustering_uid: unique identifier
    method: type of clustering performed, either PhenoGraph or FlowSOM
    parameters: parameters passed to clustering algorithm
    features: list of channels/markers that clustering is performed on
    transform_method: type of transformation to be applied to data prior to clustering
    root_population: population that clustering is performed on (default = 'root')
    cluster_prefix: a prefix to add to the name of each resulting cluster
    meta_clustering: refers to whether the clustering is 'meta-clustering'
    """
    clustering_uid = mongoengine.StringField(required=True, unique=True)
    method = mongoengine.StringField(required=True, choices=['PhenoGraph', 'FlowSOM'])
    parameters = mongoengine.ListField(required=True)
    features = mongoengine.ListField(required=True)
    transform_method = mongoengine.StringField(required=False, default='logicle')
    root_population = mongoengine.StringField(required=True, default='root')
    cluster_prefix = mongoengine.StringField(required=True, default='cluster')
    meta_clustering = mongoengine.BooleanField(required=True, default=False)
