import mongoengine


def global_init():
    """
    MongoDB setup for immunova database
    :return:
    """
    mongoengine.register_connection(alias='core', name='immunova')


def test_init():
    """
    MongoDB setup for immunova database
    :return:
    """
    mongoengine.register_connection(alias='core', name='test_server')