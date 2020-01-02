import mongoengine

def pd_init():
    mongoengine.register_connection(alias='core', name='pd_server')

def global_init(database_name: str) -> None:
    """
    Global initializer for mongogengine ORM
    :param database_name: name of database to establish connection with
    :return:
    """
    mongoengine.register_connection(alias='core', name=database_name)