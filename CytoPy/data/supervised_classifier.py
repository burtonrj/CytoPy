import mongoengine


class Classifier(mongoengine.document):
    klass = mongoengine.StringField(required=True)
    params = mongoengine.ListField()
    features = mongoengine.ListField()
    multi_label = mongoengine.BooleanField(default=True)
    transform = mongoengine.StringField(default="logicle")
    threshold = mongoengine.FloatField(default=0.5)
    scale = mongoengine.StringField()
    scale_kwargs = mongoengine.ListField()
    balance_dict = mongoengine.ListField()
    downsample = mongoengine.StringField()
    downsample_kwargs = mongoengine.ListField()

