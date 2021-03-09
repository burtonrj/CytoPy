#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
This module contains global_init which establishes a connection
to the database and should be called at the start of each script.

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
import mongoengine

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def global_init(database_name: str,
                **kwargs) -> None:
    """
    Global initializer for mongogengine ORM. See mongoengine.register_connection for additional keyword arguments and
    mongoengine documentation for extensive details about registering connections. In brief, database connections are
    registered globally and refered to using an alias. By default cytopy uses the alias 'core'.

    The database is assumed to be hosted locally, but if a remote server is used the user should provide the host
    address and port. If authentication is needed then a username and password should also be provided.

    Parameters
    -----------
    database_name: str
        name of database to establish connection with
    kwargs:
        Additional keyword arguments passed to 'register_connection' function of mongoengine.
        See https://docs.mongoengine.org/guide/connecting.html

    Returns
    --------
    None
    """
    mongoengine.register_connection(alias="core", name=database_name, **kwargs)
