from .experiment import Experiment
from .fcs import FileGroup
from .subject import Subject
from _warnings import warn


def fetch_subject_meta(sample_id: str,
                       experiment: Experiment,
                       meta_label: str):
    """
    Fetch the Subject document through a reverse search of
    associated FileGroup and return the requested meta-label
    stored in the Subject. If no Subject is found or no
    meta-label matches the search, will return None

    Parameters
    ----------
    experiment: Experiment
        Experiment containing the FileGroup of interest
    sample_id: str
        FileGroup primary ID
    meta_label: str
        Meta variable to fetch

    Returns
    -------
    Subject or None
    """
    fg = experiment.get_sample(sample_id=sample_id)
    subject = fetch_subject(filegroup=fg)
    try:
        return subject[meta_label]
    except KeyError:
        return None


def fetch_subject(filegroup: FileGroup):
    """
    Reverse search for Subject document using a FileGroup

    Parameters
    ----------
    filegroup: FileGroup

    Returns
    -------
    Subject or None
    """
    subject = Subject.objects(files=filegroup)
    if len(subject) != 1:
        warn("Requested sample is not associated to a Subject")
        return None
    return subject[0]