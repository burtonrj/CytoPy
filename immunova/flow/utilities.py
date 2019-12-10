from IPython import get_ipython
from tqdm import tqdm_notebook, tqdm


def which_environment():
    """
    Test if module is being executed in the Jupyter environment.
    :return:
    """
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return 'jupyter'
        if 'terminal' in ipy_str:
            return 'ipython'
    except:
        return 'terminal'


def progress_bar(x: iter, **kwargs) -> callable:
    """
    Generate a progress bar using the tqdm library. If execution environment is Jupyter, return tqdm_notebook
    otherwise used tqdm.
    :param x: some iterable to pass to tqdm function
    :param kwargs: additional keyword arguments for tqdm
    :return: tqdm or tqdm_notebook, depending on environment
    """
    if which_environment() == 'jupyter':
        return tqdm_notebook(x, **kwargs)
    return tqdm(x, **kwargs)
