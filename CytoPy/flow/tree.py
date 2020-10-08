from ..data.populations import Population
from typing import List
import anytree


def new_tree(data: dict):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """
    pass


def grow_tree(tree: dict,
              database_populations: list):
    """

    Parameters
    ----------
    tree
    database_populations

    Returns
    -------

    """
    pass


def construct_tree(populations: List[Population]):
    """

    Parameters
    ----------
    populations

    Returns
    -------

    """
    pass


def list_downstream_populations(self,
                                population: str) -> list or None:
    """For a given population find all dependencies

    Parameters
    ----------
    population : str
        population name

    Returns
    -------
    list or None
        List of populations dependent on given population

    """
    pass

def list_dependencies(self,
                      population: str) -> list:
    """
    For given population list all populations that this population depends on (upstream in the same branch)

    Parameters
    ----------
    population

    Returns
    -------
    list
    """
    pass


def list_child_populations(self,
                           population: str):
    pass