import mongoengine


def global_init(database_name: str) -> None:
    """
    Global initializer for mongogengine ORM

    Parameters
    -----------
    database_name: str
        name of database to establish connection with

    Returns
    --------
    None
    """
    mongoengine.register_connection(alias='core', name=database_name)