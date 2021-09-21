import logging
import os
import re
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import mongoengine
import pandas as pd

from .errors import PanelError
from .read_write import match_file_ext
from .read_write import read_headers

logger = logging.getLogger(__name__)


def load_template(path: str) -> pd.DataFrame:
    if match_file_ext(path=path, ext=".csv"):
        template = pd.read_csv(path)
    elif match_file_ext(path=path, ext=".xlsx") or match_file_ext(path=path, ext=".xls"):
        template = pd.read_excel(path)
    else:
        raise ValueError(f"Panel template should be a csv or excel file, not {path}.")
    required_columns = ["channel", "regex_pattern", "case_sensitive", "permutations", "name"]
    if not all([x in template.columns for x in required_columns]):
        raise KeyError(f"Template must contain columns {required_columns}.")
    unique_channels = len(template.channel.values) == template.channel.nunique()
    unique_names = len(template.name.values) == template.name.nunique()
    if not unique_channels and unique_names:
        raise RuntimeError("Duplicate channels and/or names detected in template.")
    return template


class Channel(mongoengine.EmbeddedDocument):
    channel = mongoengine.StringField(required=True)
    name = mongoengine.StringField(required=False)
    regex_pattern = mongoengine.StringField(required=False)
    case_sensitive = mongoengine.IntField(default=0)
    permutations = mongoengine.StringField(required=False)

    def query(self, x: str):
        if self.case_sensitive:
            if re.search(self.regex_pattern, x):
                return self.name if self.name else self.channel
            return None
        if re.search(self.regex_pattern, x, re.IGNORECASE):
            return self.name if self.name else self.channel
        if self.permutations:
            for p in self.permutations.split(","):
                if x == p:
                    return self.name if self.name else self.channel
        return None


class Panel(mongoengine.EmbeddedDocument):
    """
    Document representation of channel/marker definition for an experiment. A panel, once associated to an experiment
    will standardise data upon input; when an fcs file is created in the database, it will be associated to
    an experiment and the channel/marker definitions in the fcs file will be mapped to the associated panel.

    Attributes
    -----------

    """

    channels = mongoengine.EmbeddedDocumentListField(Channel)
    meta = {"db_alias": "core", "collection": "fcs_panels"}

    def list_channels(self) -> List[str]:
        return [x.name for x in self.channels]

    def query_channel(self, channel: str) -> str:
        matches = [channel_definition.query(channel) for channel_definition in self.channels]
        matches = [x for x in matches if x is not None]
        if len(matches) > 1:
            raise ValueError(f"Channel {channel} matched more than one definition in panel. Channels must be unique.")
        if len(matches) == 0:
            raise ValueError(f"No matching channel found in panel for {channel}")
        return matches[0]

    def build_mappings(self, path: Union[str, List[str]], s3_bucket: Optional[str] = None) -> Dict[str, str]:
        try:
            if isinstance(path, str):
                path = [path]
            columns = None
            for path in path:
                if columns is None:
                    columns = read_headers(path=path, s3_bucket=s3_bucket)
                else:
                    assert set(columns) == set(read_headers(path=path, s3_bucket=s3_bucket))
            mappings = dict()
            for col in [x for x in columns if x != "Index"]:
                mappings[col] = self.query_channel(channel=col)
            return mappings
        except AssertionError:
            err = "Columns must be identical for all files with related mappings"
            logger.error(err)
            raise ValueError(err)

    def create_from_tabular(self, path: str):
        logger.info(f"Generating new Panel definition from Excel file template {path}")
        if not os.path.isfile(path):
            raise PanelError(f"No such file {path}")
        template = load_template(path=path).fillna("").to_dict(orient="records")
        for row in template:
            self.channels.append(
                Channel(
                    channel=row["channel"],
                    regex_pattern=row["regex_pattern"],
                    case_sensitive=row["case_sensitive"],
                    permutations=row["permutations"],
                    name=row["name"],
                )
            )
