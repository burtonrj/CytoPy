from data.fcs_experiments import FCSExperiment, Panel, NormalisedName, ChannelMap
from data.project import Project


def create_fcs_experiment(project_name: str, experiment_id: str, panel_name: str):
    """
    Create a new fcs experiment
    :param project_name: name of parent project
    :param experiment_id: name of the new experiment
    :param panel_name: panel to associate to experiment
    :return: MongoDB document ID
    """
    if FCSExperiment.objects(experiment_id=experiment_id):
        print(f'Experiment with id {experiment_id} already exists!')
        return None
    if not Project.objects(project_id=project_name):
        print(f'Project {project_name} does not exist!')
        return None
    panel = Panel.objects(panel_name=panel_name)
    if not panel:
        print(f'Panel {panel_name} does not exist')
        return None
    exp = FCSExperiment()
    exp.experiment_id = experiment_id
    exp.panel = panel[0]
    exp.save()
    Project.objects(project_id=project_name).update(push__fcs_experiments=exp)
    print(f'Experiment created successfully: {exp.id.__str__()}')
    return exp.id.__str__()
