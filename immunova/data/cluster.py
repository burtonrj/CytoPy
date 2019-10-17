import mongoengine


class Cluster(mongoengine.Document):
    cluster_id = mongoengine.StringField(required=True)
    method = mongoengine.StringField(required=True)
    root_population = mongoengine.StringField(required=True, default='root')
    index = mongoengine.FileField(db_alias='core', collection_name='cluster_indexes')

