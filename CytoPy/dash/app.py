from ..dash import db_manager


def launch(app_name: str, **kwargs):
    if app_name == "db_manager":
        db_manager.APP.run_server(**kwargs)
