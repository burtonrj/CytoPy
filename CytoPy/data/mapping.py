#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Each staining panel will have a combination of detection channel and the
marker associated with that channel. This module houses ChannelMap,
a simple class that keeps track of these mappings.

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
__copyright__ = "Copyright 2020, CytoPy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"


class ChannelMap(mongoengine.EmbeddedDocument):
    """
    Defines channel/marker mapping. Each document will contain a single value for channel and a single value for marker,
    these two values are treated as a pair within the panel.

    Attributes
    ----------
    channel: str
        name of channel (e.g. fluorochrome)
    marker: str
        name of marker (e.g. protein)
    """
    channel = mongoengine.StringField()
    marker = mongoengine.StringField()

    def check_matched_pair(self, channel: str or None, marker: str or None) -> bool:
        """
        Check a channel/marker pair for resemblance

        Parameters
        ----------
        channel: str
            channel to check
        marker: str
            marker to check

        Returns
        --------
        bool
            True if equal, else False
        """
        channel = channel or ""
        marker = marker or ""
        if self.channel == channel and self.marker == marker:
            return True
        return False

    def to_dict(self) -> dict:
        """
        Convert object to python dictionary

        Returns
        --------
        dict
        """
        return {'channel': self.channel, 'marker': self.marker}