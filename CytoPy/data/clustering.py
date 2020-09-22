from .experiments import Experiment
import mongoengine


class ClusteringDefinition(mongoengine.Document):
    name = mongoengine.StringField(required=True, unique=True)
    features = mongoengine.ListField(required=True)
    transform_method = mongoengine.StringField(required=False, default="logicle")
    root_population = mongoengine.StringField(required=True, default="root")
    experiment = mongoengine.ReferenceField(Experiment, reverse_delete_rule=mongoengine.CASCADE)
    method = mongoengine.StringField(required=True, choices=["PhenoGraph", "FlowSOM"])
    kwargs = mongoengine.DictField()
    prefix = mongoengine.StringField(default="cluster")

    meta = {
        "db_alias": "core",
        "collection": "clustering_experiments"
    }

