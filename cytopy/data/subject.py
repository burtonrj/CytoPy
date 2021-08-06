#!/usr/bin.env/python
# -*- coding: utf-8 -*-
"""
Each subject in your analysis (human, mouse, cell line etc) can be
represented by a Subject document that can then be associated to
specimens in an Experiment. This Subject document is dynamic and can
house any relating meta-data.

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
from typing import Dict
from typing import List
from typing import Set

import mongoengine
import pandas as pd

__author__ = "Ross Burton"
__copyright__ = "Copyright 2020, cytopy"
__credits__ = ["Ross Burton", "Simone Cuff", "Andreas Artemiou", "Matthias Eberl"]
__license__ = "MIT"
__version__ = "2.0.0"
__maintainer__ = "Ross Burton"
__email__ = "burtonrj@cardiff.ac.uk"
__status__ = "Production"

logger = logging.getLogger(__name__)


class Subject(mongoengine.DynamicDocument):
    """
    Document based representation of subject meta-data.
    Subjects are stored in a dynamic document, meaning
    new properties can be added ad-hoc.

    Attributes
    -----------
    subject_id: str, required
        Unique identifier for subject
    notes: str
        Additional notes
    """

    subject_id = mongoengine.StringField(required=True, unique=True)
    notes = mongoengine.StringField(required=False)

    meta = {"db_alias": "core", "collection": "subjects"}

    @property
    def fields(self) -> List[str]:
        return list(self.to_dict().keys())

    @fields.setter
    def fields(self, _):
        raise ValueError("Fields is read only, access individual fields to edit values.")

    def to_dict(self, *args, **kwargs) -> Dict:
        return json.loads(self.to_json(*args, **kwargs))

    def field_to_df(self, field: str, **kwargs) -> pd.DataFrame:
        if len(self.__getitem__(field)) == 0:
            logger.warning(f"filed '{field}' is empty")
            return None
        try:
            return pd.DataFrame(self.to_dict()[field], **kwargs)
        except KeyError:
            logger.error(f"{field} is not a recognised field")
        except ValueError:
            return (
                pd.DataFrame(self.to_dict()[field], index=["value"], **kwargs)
                .T.reset_index()
                .rename({"index": field}, axis=1)
            )


def common_fields(subjects: List[Subject]) -> Set:
    return set.intersection(*[set(s.fields) for s in subjects])


def safe_search(subject_id: str):
    try:
        return Subject.objects(subject_id=subject_id).get()
    except mongoengine.DoesNotExist:
        return None
