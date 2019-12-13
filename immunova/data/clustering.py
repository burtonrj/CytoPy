import mongoengine


class ClusteringExperiment(mongoengine.Document):
    clustering_uid = mongoengine.StringField(required=True)
    method = mongoengine.StringField(required=True)
    parameters = mongoengine.ListField(required=True)
    features = mongoengine.ListField(required=False)
    transform_method = mongoengine.StringField(required=False)
    root_population = mongoengine.StringField(required=True)
    cluster_prefix = mongoengine.StringField(required=False)