assert os.path.isfile(panel_definition), f"{panel_definition} doe not exist"
err = "Panel definition is not a valid Excel document"
assert os.path.splitext(panel_definition)[1] in [".xls", ".xlsx"], err