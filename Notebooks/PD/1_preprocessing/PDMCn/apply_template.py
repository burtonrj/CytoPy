from immunova.data.fcs_experiments import FCSExperiment
from immunova.flow.gating.actions import Template

def load_data(experiment_id, sample_id, template_id):
    exp = FCSExperiment.objects(experiment_id=experiment_id).get()
    t = Template(exp, sample_id, include_controls=False)
    t.clear_gates()
    t.load_template(template_id)
    return t
    
def apply_n_template(sample_id, template_id='PDMCn_Preprocessing_Secondary'):
    t = load_data(experiment_id='PD_N_PDMCs', sample_id=sample_id, template_id=template_id)
    t.apply_many(apply_all=True, plot_outcome=True)
    t.plotting.plot_population('Single_Live_CD45+', 'FSC-A', 'SSC-A', transforms={'x': None, 'y': None})
    return t



    