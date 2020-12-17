import pandas as pd


def assert_fitted(func):
    def wrapper(*args, **kwargs):
        assert len(args[0].standard_curves) != 0, "Standard curves have not been computed; call 'fit' prior to " \
                                                  "additional functions"
        if "analyte" in kwargs.keys():
            assert kwargs.get("analyte") in args[0].standard_curves.keys(),\
                f"Standard curve not detected for {kwargs.get('analyte')}; call 'fit' prior to additional functions"
        return func(*args, **kwargs)
    return wrapper


def subtract_background(data: pd.DataFrame,
                        background_id: str,
                        analytes: list):
    """
    Given a DataFrame of assay data (as described in the documentation for AssayTools), subtract
    background using the rows with a Sample ID corresponding to the 'background_id'.

    Parameters
    ----------
    data: Pandas.DataFrame
    background_id: str
    analytes: list
        List of columns corresponding to analytes

    Returns
    -------
    Pandas.DataFrame
    """
    data = data.copy()
    assert background_id in data["Sample"].values, "Invalid background_id, not found in Sample column"
    background = data[data.Sample == background_id].copy()
    data = data[~data.index.isin(background.index)].copy()
    for analyte, mean_ in background.mean().to_dict().items():
        data[analyte] = data[analyte] - mean_
        data[analyte] = data[analyte].apply(lambda x: x if x > 0 else 0)
    return data


class AssayTools:
    """
    Tools for analysis of plate based assay data such as ELISAs and Luminex assays.
    Calculate standard curves, predict concentrations, transform data, as well as
    access to various plotting functions and statistical tests.

    Attributes
    ----------
    raw: Pandas.DataFrame
        Raw unaltered assay data. Will also contain any associated meta data if the subject ID is
        provided.
    predictions: Pandas.DataFrame
        Predicted concentrations of analytes using standard curves
    analytes: list
        List of analytes being studied
    concentrations: Pandas.DataFrame
        Concentrations corresponding to each standard control
    standard_curves: dict
        Fitted functions for each analyte
    """
    def __init__(self,
                 data: pd.DataFrame,
                 conc: pd.DataFrame,
                 standards: list,
                 background_id: str or None = None,
                 analytes: list or None = None):
        """
        Parameters
        ----------
        data: Pandas.DataFrame
            Assay data. Must contain the column "Sample" with each value being an individual biological
            sample; include duplicates as individual rows. Standards should be included as samples, so
            with each standard in its own row; again, including duplicates. Additional columns should
            correspond to analytes. The user can also include an additional row called "subject_id"
            for matching rows to meta data.
        conc: Pandas.DataFrame
            Concentrations of each analyte for each standard control. There should be one column named "analyte"
            with a unique row for each analyte. The remaining columns should correspond to the standard controls.
            Standards will be interpreted in the order given in the "standards" parameter.
        standards: list
            List of unique standard names. Order matters! Standards will be treated in this order as highest
            concentration to lowest.
        background_id: str, optional
            Sample ID corresponding to rows to be used for subtracting background.
        analytes: list, optional
            List of analytes measured. If not given, all columns in data not equal to "Sample" or "subject_id"
            will be treated as analytes.
        """

        self.analytes = analytes or [x for x in data.columns if x not in ["Sample", "subject_id"]]
        assert "Sample" in data.columns, "Invalid DataFrame missing 'Sample' column"
        assert all([x in data.columns for x in self.analytes]), "One or more listed analytes missing from data"
        assert all([x in data["Sample"].values for x in standards]), \
            "One or more listed standards missing from Sample column"
        assert "analyte" in conc.columns, "Analyte column missing from concentrations dataframe"
        assert all([x in conc.columns for x in standards]), \
            "One or more of the specified standards missing from concentrations dataframe"
        assert all([x in conc["analyte"].values for x in self.analytes]), \
            "One or more of the specified analytes missing from concentrations dataframe"
        self.concentrations = conc
        self.standard_curves = dict()
        self._predictions = dict()
        self.standards = standards
        self.raw = data
        if background_id:
            self.raw = subtract_background(data=data,
                                           background_id=background_id,
                                           analytes=self.analytes)


    @property
    def predictions(self):
        x = pd.DataFrame(self._predictions)
        for c in self.raw.columns:
            if c not in self._predictions.keys():
                x[c] = self.raw[c]
        return x

    @predictions.setter
    def predictions(self, _):
        raise ValueError("Predictions is a read-only property. Call fit_predict to fit standard curves and "
                         "popualte predictions.")

    def _validate_data(self):
        pass

    def _validate_conc(self):
        pass

    def fit(self,
            func: callable,
            analyte: str or None = None):
        pass

    @assert_fitted
    def predict(self,
                analyte: str or None = None):
        pass

    def fit_predict(self,
                    func: callable,
                    analyte: str or None = None):
        pass

    def plot_intensity(self,
                       transform: str = "norm"):
        pass

    @assert_fitted
    def plot_standard_curve(self,
                            analyte: str,
                            overlay_standards: bool = True,
                            overlay_data: bool = True,
                            conf_interval: bool = True):
        # https://pingouin-stats.org/generated/pingouin.compute_bootci.html#pingouin.compute_bootci
        pass

    @assert_fitted
    def plot_repeat_measures(self):
        pass
        # https://pingouin-stats.org/generated/pingouin.plot_paired.html#pingouin.plot_paired

    @assert_fitted
    def plot_shift(self,
                   analyte: str,
                   factor: str):
        # https://pingouin-stats.org/generated/pingouin.plot_shift.html#pingouin.plot_shift
        pass

    @assert_fitted
    def corr_matrix(self):
        pass

    @assert_fitted
    def plot_box_swarm(self,
                       analyte: str,
                       factor: str):
        pass

    @assert_fitted
    def volcano_plot(self,
                     factor: str,
                     stat: str or None = None,
                     eff_size: str or None = None):
        pass

    def load_meta(self):
        pass

    @assert_fitted
    def statistics(self,
                   factor: str):
        pass

