class MissingExperimentError(Exception):
    pass


class DuplicateExperimentError(Exception):
    pass


class DuplicateSubjectError(Exception):
    pass


class DuplicateSampleError(Exception):
    pass


class DuplicatePopulationError(Exception):
    pass


class InvalidDataDirectory(Exception):
    pass


class MissingSubjectError(Exception):
    pass


class MissingSampleError(Exception):
    pass


class MissingControlError(Exception):
    """Raised when control file missing from FileGroup"""
    pass


class MissingPopulationError(Exception):
    """Raised when population requested is missing from FileGroup"""
