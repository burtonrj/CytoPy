from ...data.experiments import check_excel_template
from ...tests import assets
import pandas as pd
import pytest
import os


def test_check_excel_template():
    mappings, nomenclature = check_excel_template(f"{assets.__path__._path[0]}/test_panel.xlsx")
    assert isinstance(mappings, pd.DataFrame)
    assert isinstance(nomenclature, pd.DataFrame)

