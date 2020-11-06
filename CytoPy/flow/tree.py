#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
CytoPy tracks the population "tree" of a FileGroup when a FileGroup
is loaded into memory and is being analysed. This module handles the
creation and modification of this "tree" using the anytree library.

Copyright 2020 Ross Burton

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from ..data.population import Population
from typing import List, Dict
import anytree

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def _add_leaf(tree: Dict[str, anytree.Node],
              population: Population):
    """
    Add a new 'leaf' (node) to the population tree (represented by a dictionary of
    anytree Node objects).

    Parameters
    ----------
    tree: dict
        {population name: anytree.Node}
    population: Population
        Population to add to the tree

    Returns
    -------
    dict
    """

    if population.parent not in tree.keys():
        return None
    tree[population.population_name] = anytree.Node(name=population.population_name,
                                                    parent=tree[population.parent])
    return tree


def _grow_tree(tree: Dict[str, anytree.Node],
               database_populations: List[Population]):
    """
    Given a list of Population objects, grow the 'tree' (represented by a dictionary of
    anytree Node objects) according to the 'parent' attribute of each population.

    Parameters
    ----------
    tree: dict
        {population name: anytree.Node}
    database_populations: list
        List of Populations to add to the tree

    Returns
    -------

    """
    i = 0
    while len(database_populations) > 0:
        if i >= len(database_populations):
            # Loop back around
            i = 0
        branch = _add_leaf(tree, database_populations[i])
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

