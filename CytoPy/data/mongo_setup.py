from dask.distributed import Client
from multiprocessing import cpu_count
import mongoengine


def global_init(database_name: str,
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

    Returns
    --------
    None
    """
    mongoengine.register_connection(alias='core', name=database_name, **kwargs)


def dask_client(n_cores: int or None = None,
                **kwargs):
    """
    Setup Dask client for session.

    Parameters
    ----------
    n_cores: int (optional)
        Number of processes to use (if not provided, will use all available processes)
    kwargs:
        Additional keyword arguments to pass to dask.distributed.Client call

    Returns
    -------
    dask.distributed.Client
    """
    if n_cores is None:
        n_cores = cpu_count()
    return Client(n_workers=n_cores, **kwargs)


class Settings(mongoengine.Document):
    _data_directory = mongoengine.StringField(db_field="data_directory")

    def __init__(self, *args, **kwargs):
        if "data_directory" in kwargs.keys():
            self.definition = kwargs.pop("definition")
        super().__init__(*args, **kwargs)
