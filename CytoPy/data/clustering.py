import mongoengine


class ClusteringDefinition(mongoengine.Document):
    name = mongoengine.StringField(required=True, unique=True)
    features = mongoengine.ListField(required=True)
    transform_method = mongoengine.StringField(required=False, default="logicle")
    root_population = mongoengine.StringField(required=True, default="root")
    method = mongoengine.StringField(required=True, choices=["PhenoGraph", "FlowSOM"])
    kwargs = mongoengine.DictField()
    prefix = mongoengine.StringField(default="cluster")

    meta = {
        "db_alias": "core",
        "collection": "clustering_definitions"
    }


class MetaClusteringDefinition:
    name = mongoengine.StringField(required=True, unique=True)
    target = mongoengine.StringField(required=True)
    transform_method = mongoengine.StringField(required=False, default="logicle")
    norm_method = mongoengine.StringField(required=True, default="norm", choices=[None, "norm", "standardise"])
    summary_method = mongoengine.StringField(required=True, default="median", choices=["mean", "median"])
    cluster_method = mongoengine.StringField(required=True, default="PhenoGraph", choices=["PhenoGraph", "FlowSOM",
                                                                                           "ConsensusClustering", "Agglomerative"])
    kwargs = mongoengine.DictField()
    prefix = mongoengine.StringField(default="meta")

    meta = {
        "db_alias": "core",
        "collection": "meta_clustering_definitions"
    }


