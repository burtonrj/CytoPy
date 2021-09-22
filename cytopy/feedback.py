import queue
import sys
import threading
import time
from typing import Callable

from IPython import get_ipython
from tqdm import tqdm
from tqdm import tqdm_notebook


def processing_animation(text: str):
    for c in ["|", "/", "-", "\\"]:
        sys.stdout.write(f"\r{text} {c}\r")
        time.sleep(0.1)
        sys.stdout.flush()


def add_processing_animation(text: str = "processing") -> Callable:
    def wrapper(func: Callable):
        q = queue.Queue()

        def store_in_queue(*args, **kwargs):
            q.put(func(*args, **kwargs))

        def wrap_with_ani(*args, **kwargs):
            process = threading.Thread(target=store_in_queue, args=args, kwargs=kwargs)
            process.start()
            while process.is_alive():
                processing_animation(text=text)
            sys.stdout.flush()
            sys.stdout.write(f"\rDone!{' ' * (len(text) + 3)}")
            return q.get()

        return wrap_with_ani

    return wrapper


def progress_bar(x: iter, verbose: bool = True, **kwargs) -> callable:
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
    if which_environment() == "jupyter":
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
        if "zmqshell" in ipy_str:
            return "jupyter"
        if "terminal" in ipy_str:
            return "ipython"
    except:
        return "terminal"


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
