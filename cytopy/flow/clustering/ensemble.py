from ...data.experiment import Experiment
from typing import *
import logging

logger = logging.getLogger("clustering.ensemble")

class EnsembleClustering:
    def __init__(self,
                 experiment: Experiment,
                 sample_size: Optional[int, float]):
        pass