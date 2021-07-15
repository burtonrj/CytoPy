import logging
logger = logging.getLogger("data")


class DataError(Exception):
    def __init__(self, message: str):
        logger.error(message)
        super().__init__(message)


class MissingExperimentError(DataError):
    def __init__(self, experiment_name: str):
        super(self).__init__(f"Invalid experiment, {experiment_name} does not exist!")


class DuplicateExperimentError(Exception):
    def __init__(self, experiment_name: str):
        super(self).__init__(f"Invalid experiment, {experiment_name} already exists!")


class DuplicateSubjectError(Exception):
    def __init__(self, subject_id: str):
        super(self).__init__(f"Invalid subject, {subject_id} already exists!")


class DuplicateSampleError(Exception):
    def __init__(self, sample_id: str):
        super(self).__init__(f"Invalid sample ID, {sample_id} already exists!")


class DuplicatePopulationError(Exception):
    def __init__(self, population_id: str):
        super(self).__init__(f"Invalid population, {population_id} already exists!")


class DuplicateGateError(Exception):
    def __init__(self, gate_id: str):
        super(self).__init__(f"Invalid gate, {gate_id} already exists!")


class InvalidDataDirectory(Exception):
    def __init__(self, path: str):
        super(self).__init__(f"Invalid directory, {path} does not exists!")


class MissingSubjectError(Exception):
    def __init__(self, subject_id: str):
        super(self).__init__(f"Invalid subject, {subject_id} does not exists!")


class MissingSampleError(Exception):
    def __init__(self, sample_id: str):
        super(self).__init__(f"Invalid FileGroup, {sample_id} does not exists!")


class MissingControlError(Exception):
    def __init__(self, ctrl: str):
        super(self).__init__(f"Invalid control, {ctrl} does not exists!")


class MissingPopulationError(Exception):
    def __init__(self, population_id: str):
        super(self).__init__(f"Invalid population, {population_id} does not exists!")


class InsufficientEventsError(Exception):
    def __init__(self,
                 population_id: str,
                 filegroup_id: str):
        super(self).__init__(f"Insufficient events in {population_id} does not exists for {filegroup_id}!")

