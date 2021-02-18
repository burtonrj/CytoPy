from IPython import get_ipython
from tqdm import tqdm_notebook, tqdm
import logging


def setup_standard_logger(name: str,
                          default_level: int or None = None,
                          log: str or None = None):
    default_level = default_level or logging.INFO
    logger = logging.getLogger(name)
    logger.setLevel(default_level)
    if log is not None:
        handler = logging.FileHandler(filename=log)
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger


def progress_bar(x: iter,
                 verbose: bool = True,
                 **kwargs) -> callable:
    """
    Generate a progress bar using the tqdm library. If execution environment is Jupyter, return tqdm_notebook
    otherwise used tqdm.

    Parameters
    -----------
    x: iterable
        some iterable to pass to tqdm function
    verbose: bool, (default=True)
        Provide feedback (if False, no progress bar produced)
    kwargs:
        additional keyword arguments for tqdm
    :return: tqdm or tqdm_notebook, depending on environment
    """
    if not verbose:
        return x
    if which_environment() == 'jupyter':
        return tqdm_notebook(x, **kwargs)
    return tqdm(x, **kwargs)


def which_environment() -> str:
    """
    Test if module is being executed in the Jupyter environment.

    Returns
    -------
    str
        'jupyter', 'ipython' or 'terminal'
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


def vprint(verbose: bool):
    """
    Utility function for optional printing.

    Parameters
    ----------
    verbose: bool
        If True, returns print function, else False
    Returns
    -------
    callable
    """
    return print if verbose else lambda *a, **k: None
