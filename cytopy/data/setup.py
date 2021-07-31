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
from logging.handlers import RotatingFileHandler
from typing import *
import mongoengine
import logging
import cytopy
import json
import os

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


def setup_logs(path: Optional[str] = None,
               level: int = logging.INFO) -> None:
    """
    Setup logging

    Parameters
    ----------
    path: str, optional
        Where to store logs (defaults to home path)
    level: int (default=logging.INFO)

    Returns
    -------
    None
    """
    path = path or "cytopy.log"
    logging.basicConfig(level=level,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[
                            RotatingFileHandler(path, maxBytes=2e+6, backupCount=10),
                            logging.StreamHandler()
                        ])


def global_init(database_name: str,
                logging_path: Optional[str] = None,
                logging_level: int = logging.INFO,
                **kwargs) -> None:
    """
    Global initializer for mongogengine ORM and logging. Logging is managed using the loguru package.

    See mongoengine.register_connection for additional keyword arguments and mongoengine
    documentation for extensive details about registering connections. In brief, database connections are
    registered globally and referred to using an alias. By default cytopy uses the alias 'core'.

    The database is assumed to be hosted locally, but if a remote server is used the user should provide the host
    address and port. If authentication is needed then a username and password should also be provided.

    Parameters
    -----------
    database_name: str
        name of database to establish connection with
    logging_path: Union[str, None]
        defaults to home path
    logging_level: int (default=logging.INFO)
    kwargs:
        Additional keyword arguments passed to 'register_connection' function of mongoengine.
        See https://docs.mongoengine.org/guide/connecting.html

    Returns
    --------
    None
    """
    setup_logs(logging_path, level=logging_level)
    mongoengine.register_connection(alias="core", name=database_name, **kwargs)
    logging.info(f"Environment setup. Logging to {logging_path or 'cytopy.log'}. "
                 f"Connected to {database_name} database.")


class Config:
    def __init__(self,
                 path: Optional[str] = None):
        cytopy_path = os.path.dirname(cytopy.__file__)
        path = path or os.path.join(cytopy_path, "config.json")
        with open(path, "r") as f:
            self.options = json.load(f)

    def __getitem__(self, item: str):
        try:
            return self.options[item]
        except KeyError:
            raise KeyError(f"Invalid option, valid options are: {self.options.keys()}")

    def __getattr__(self, item: str):
        self.__getitem__(item=item)

    def __str__(self):
        return json.dumps(self.options, indent=4)

    def save(self,
             path: Optional[str] = None):
        with open(path, "w") as f:
            json.dump(self.options, f)
