import mongoengine


class Explorer(mongoengine.Document):
    name = mongoengine.StringField(required=True)
    transform = mongoengine.StringField(required=True)
    root_population = mongoengine.StringField(required=True)
