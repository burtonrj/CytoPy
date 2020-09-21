import mongoengine


def global_init(database_name: str,
                alias: str = "core",
                **kwargs) -> None:
    """
    Global initializer for mongogengine ORM. See mongoengine.register_connection for additional keyword arguments and
    mongoengine documentation for extensive details about registering connections. In brief, database connections are
    registered globally and refered to using an alias. By default CytoPy uses the alias 'core'.

    The database is assumed to be hosted locally, but if a remote server is used the user should provide the host
    address and port. If authentication is needed then a username and password should also be provided.

    Parameters
    -----------
    database_name: str
        name of database to establish connection with
    alias: str (default="core")
        name of connection to generate

    Returns
    --------
    None
    """
    mongoengine.register_connection(alias=alias, name=database_name, **kwargs)
