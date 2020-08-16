import mongoengine


class Classifier(mongoengine.document):
    model_name = mongoengine.StringField(required=True, unique=True)
    klass = mongoengine.StringField(required=True)
    params = mongoengine.ListField()
    features = mongoengine.ListField()
    multi_label = mongoengine.BooleanField(default=True)
    test_frac = mongoengine.FloatField(default=0.3)
    transform = mongoengine.StringField(default="logicle")
    threshold = mongoengine.FloatField(default=0.5)
    scale = mongoengine.StringField()
    scale_kwargs = mongoengine.ListField()
    balance = mongoengine.StringField()
    balance_dict = mongoengine.ListField()
    downsample = mongoengine.StringField()
    downsample_kwargs = mongoengine.ListField()
    population_prefix = mongoengine.StringField()

