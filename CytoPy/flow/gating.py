from ..data.fcs_experiments import FCSExperiment

class Gating:
    """
    Central class for performing semi-automated gating and storing gating information on an FCS FileGroup
    of a single sample.

    Parameters
    -----------
    experiment: FCSExperiment
        experiment you're currently working on
    sample_id: str
        name of the sample to analyse (must belong to experiment)
    sample: int, optional
        number of events to sample from FCS file(s) (optional)
    include_controls: bool, (default=True)
        if True and FMOs are included for specified samples, the FMO data will also be loaded into the Gating object
    """

    def __init__(self,
                 experiment: FCSExperiment,
                 sample_id: str,
                 sample: int or None = None,
                 include_controls=True):
        data = experiment.pull_sample_data(sample_id=sample_id,
                                           sample_size=sample,
                                           include_controls=include_controls)
        assert data is not None, f'Error: failed to fetch data for {sample_id}. Aborting.'
