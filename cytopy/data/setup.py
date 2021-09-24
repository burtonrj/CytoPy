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
import json
import logging
import os
from logging.config import dictConfig
from types import SimpleNamespace
from typing import *

import mongoengine

import cytopy


class Config(SimpleNamespace):
    """
    Configuration handler. Configurations are stored as a JSON file in the CytoPy installation
    directory and can be accessed using this class. Once initialised, configurations take the form
    of a standard dictionary. The first level of keys in the configuration can be accessed like
    attributes using dot notation.

    To modify configurations, call the 'save' method, which will overwrite configurations using the
    current state.

    Parameters
    ----------
    path: str, optional
        Use for testing purposes. Bypasses the default config file and will use the given JSON instead.
    """

    def __init__(self, path: Optional[str] = None):
        self.install_path = os.path.dirname(cytopy.__file__)
        path = path or os.path.join(self.install_path, "config.json")
        with open(path, "r") as f:
            super().__init__(**json.load(f))

    def __getitem__(self, item: str):
        try:
            return self.__dict__[item]
        except KeyError:
            raise KeyError(f"Invalid option, valid options are: {self.options.keys()}")

    def get(self, item: str, default=None):
        try:
            self.__getitem__(item=item)
        except KeyError:
            return default

    def __str__(self):
        return json.dumps(self.__dict__, indent=4)

    def update_logger_level(self, handler_id: str, level: Union[str, int]):
        levels = {
            "CRITICAL": 50,
            "ERROR": 40,
            "WARNING": 30,
            "INFO": 20,
            "DEBUG": 10,
            "NOTSET": 0,
        }
        try:
            level = level if isinstance(level, int) else levels[level.upper()]
        except KeyError:
            raise KeyError(f"{level} is not a valid level, must be one of {levels.keys()}")

        try:
            self.__getitem__("logging_config")["handlers"][handler_id] = level
        except KeyError:
            raise KeyError("Logging config not defined or invalid handler")

    def save(self, path: Optional[str] = None):
        with open(path, "w") as f:
            json.dump(self.options, f)


def global_init(database_name: str, config_path: Optional[str] = None, **kwargs) -> None:
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
        Name of database to establish connection with.
    config_path: str, optional
        Overwrite default configuration file. WARNING: must follow configuration template.
    kwargs:
        Additional keyword arguments passed to 'register_connection' function of mongoengine.
        See https://docs.mongoengine.org/guide/connecting.html

    Returns
    --------
    None
    """
    config = Config(path=config_path)
    dictConfig(config.logging_config)
    logger = logging.getLogger(__name__)
    mongoengine.register_connection(alias="core", name=database_name, **kwargs)
    logger.info(f"Connected to {database_name} database.")
