import mongoengine


class Classifier(mongoengine.Document):
    model_name = mongoengine.StringField(required=True, unique=True)
    features = mongoengine.ListField()
    multi_label = mongoengine.BooleanField(default=True)
    transform = mongoengine.StringField(default="logicle")
    scale = mongoengine.StringField()
    scale_kwargs = mongoengine.ListField()
    balance = mongoengine.StringField()
    balance_dict = mongoengine.ListField()
    downsample = mongoengine.StringField()
    downsample_kwargs = mongoengine.ListField()
    population_prefix = mongoengine.StringField()


class SklearnClassifier(Classifier):
    klass = mongoengine.StringField(required=True)
    params = mongoengine.DictField()

    def save(self, *args, **kwargs):
        if self.multi_label:
            assert self.klass in ["DescisionTreeClassifier",
                                  "ExtraTreeClassifier",
                                  "ExtraTreesClassifier",
                                  "KNeighborsClassifier",
                                  "MLPClassifier",
                                  "RadiusNeighborsClassifier",
                                  "RandomForestClassifier",
                                  "RidgeClassifierCV"]
        super().save(*args, **kwargs)


class Layer(mongoengine.EmbeddedDocument):
    klass = mongoengine.StringField()
    kwargs = mongoengine.DictField()


class KerasClassifier(Classifier):
    model_params = mongoengine.StringField()
    input_layer = mongoengine.EmbeddedDocumentField(Layer)
    layers = mongoengine.EmbeddedDocumentListField(Layer)
    optimizer = mongoengine.StringField()
    loss = mongoengine.StringField()
    metrics = mongoengine.ListField()
    epochs = mongoengine.IntField()
    compile_kwargs = mongoengine.DictField()

