from ..data.population import Population
from typing import List, Dict
import anytree


def _construct_branch(tree: Dict[str, anytree.Node],
                      population: Population):
    """

    Parameters
    ----------
    tree
    population

    Returns
    -------

    """

    if population.parent not in tree.keys():
        return None
    tree[population.population_name] = anytree.Node(name=population.population_name,
                                                    parent=tree[population.parent])
    return tree


def _grow_tree(tree: Dict[str, anytree.Node],
               database_populations: List[Population]):
    """

    Parameters
    ----------
    tree
    database_populations

    Returns
    -------

    """
    i = 0
    while len(database_populations) > 0:
        if i >= len(database_populations):
            # Loop back around
            i = 0
        branch = _construct_branch(tree, database_populations[i])
        if branch is not None:
            tree = branch
            database_populations = [p for p in database_populations
                                    if p.population_name != database_populations[i].population_name]
        else:
            i = i + 1
    return tree


def construct_tree(populations: List[Population]) -> Dict[str, anytree.Node]:
    """
    Given a list of populations, construct a tree of population hierarchy using the
    population parent information.

    Parameters
    ----------
    populations: List[Population]
        List of Population objects
    Returns
    -------
    dict
        Dictionary of Node objects
    """
    err = "Invalid FileGroup, must contain 'root' population"
    assert "root" in [p.population_name for p in populations], err
    tree = {"root": anytree.Node(name="root", parent=None)}
    database_populations = [p for p in populations if p.population_name != 'root']
    return _grow_tree(tree=tree, database_populations=database_populations)


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

