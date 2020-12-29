from ..feedback import progress_bar
from .transform import apply_transform
from warnings import warn
from lmfit import Model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def linear(x: np.ndarray,
           coef: float,
           bias: float):
    """
    Function to fit a line to data points 'x'

    Parameters
    ----------
    x: Numpy.Array
    coef: float
        Slope or 'coefficient' of the 'curve' (straight line)
    bias: float
        Bias term (i.e the error or noise)

    Returns
    -------
    Numpy.Array
    """
    return coef * x + bias


def poly2(x: np.ndarray,
          coef1: float,
          coef2: float,
          bias: float):
    """
    Second order polynomial function

    Parameters
    ----------
    x: Numpy.Array
    coef1: float
        First coefficient or "slope"
    coef2: float
        Second coefficient
    bias: float
        Bias term (i.e the error or noise)

    Returns
    -------
    Numpy.Array
    """
    return coef1 * x + coef2 * x ** 2 + bias


def poly3(x: np.ndarray,
          coef1: float,
          coef2: float,
          coef3: float,
          bias: float):
    """
    Third order polynomial function i.e. a cubic function

    Parameters
    ----------
    x: Numpy.Array
    coef1: float
        First coefficient or "slope steepness"
    coef2: float
        Second coefficient or "slope steepness"
    coef3: float
        Third coefficient or "slope steepness"
    bias: float
        Bias term (i.e the error or noise)

    Returns
    -------
    Numpy.Array
    """
    return coef1 * x + coef2 * x ** 2 + coef3 * x ** 3 + bias


def four_param_logit(x: np.ndarray,
                     min_: float,
                     max_: float,
                     inflection_point: float,
                     coef: float):
    """
    Four parameter logistic function.

    Parameters
    ----------
    x: Numpy.Array
    min_: float
        The minimum asymptote i.e. where the response value should be 0 for the standard concentration
    max_: float
        The maximum asymptote i.e. where the response value where the standard concentration is infinite
    inflection_point: float
        Where the curvature changes direction or sign
    coef: float
        The coefficient or steepness of the slope

    Returns
    -------
    Numpy.Array
    """
    n = (min_ - max_)
    d = (1 + (x / inflection_point) ** coef)
    return max_ + (n / d)


def five_param_logit(x: np.ndarray,
                     min_: float,
                     max_: float,
                     inflection_point: float,
                     coef: float,
                     asymmetry_factor: float = 1.0):
    """
    Four parameter logistic function.

    Parameters
    ----------
    x: Numpy.Array
    min_: float
        The minimum asymptote i.e. where the response value should be 0 for the standard concentration
    max_: float
        The maximum asymptote i.e. where the response value where the standard concentration is infinite
    inflection_point: float
        Where the curvature changes direction or sign
    coef: float
        The coefficient or steepness of the slope
    asymmetry_factor: float
        The asymmetry factor; where asymmetry_factor = 1, we have a symmetrical curve around the inflection
        point and so we have a four parameter logistic equation

    Returns
    -------
    Numpy.Array
    """
    n = (min_ - max_)
    d = (1 + (x / inflection_point) ** coef) ** asymmetry_factor
    return max_ + (n / d)


def assert_fitted(method):
    """
    Given a function method of AssayTools, inspect the self argument to access
    standard curves and check that the curve has been fitted. Raises assertion error if
    condition is not met.

    Parameters
    ----------
    method: callable

    Returns
    -------
    callable
        Original method

    Raises
    ------
    AssertionError
    """
    def wrapper(*args, **kwargs):
        assert len(args[0].standard_curves) != 0, "Standard curves have not been computed; call 'fit' prior to " \
                                                  "additional functions"
        if "analyte" in kwargs.keys():
            assert kwargs.get("analyte") in args[0].standard_curves.keys(), \
                f"Standard curve not detected for {kwargs.get('analyte')}; call 'fit' prior to additional functions"
        return method(*args, **kwargs)

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
    background = data[data.Sample == background_id][analytes].copy()
    data = data[~data.index.isin(background.index)].copy()
    for analyte, mean_ in background.mean().to_dict().items():
        data[analyte] = data[analyte] - mean_
        data[analyte] = data[analyte].apply(lambda x: x if x > 0 else 0.00000001)
    return data


def residuals(func: callable,
              x: np.ndarray,
              y: np.ndarray,
              params: np.ndarray):
    """
    Calculate the residuals for a given function and parameters

    Parameters
    ----------
    func: callable
        Function to be fit
    x: Numpy.Array
        Dependent variable
    y: Numpy.Array
        Independent variable
    params: Numpy.Array
        Parameters to fit

    Returns
    -------
    Numpy.Array
    """
    return y - func(x, *params)


def r_squared(err: np.ndarray,
              y: np.ndarray):
    """
    Given the residuals (error) around some given function and the
    independent variable, y, return the R-squared value

    Parameters
    ----------
    err: Numpy.Array
    y: Numpy.Array

    Returns
    -------
    float
    """
    ss_res = np.sum(err ** 2)
    ss_total = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_total)


def _fitting_functions(func: str):
    """
    Mapper that translates common name of function to function

    Parameters
    ----------
    func: str

    Returns
    -------
    callable

    Raises
    ------
    AssertionError
        Raises error if function name is unrecognised
    """
    funcs = {"linear": linear,
             "poly2": poly2,
             "poly3": poly3,
             "4pl": four_param_logit,
             "5pl": five_param_logit}
    assert func in funcs.keys(), f"{func} is an invalid function, must be one of {funcs.keys()}"
    return funcs.get(func)


INIT_PARAMS = {"linear": None,
               "poly2": None,
               "poly3": None,
               "4pl": [0, 10000, 0.5, 1],
               "5pl": [0, 10000, 0.5, 1, 0.5]}


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
        assert all([x in conc.columns.values for x in standards]), \
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
        other_vars = [c for c in self.raw.columns if c not in self.analytes]
        for c in other_vars:
            x[c] = self.raw[~self.raw.Sample.isin(self.standards)][c].values
        return x

    @predictions.setter
    def predictions(self, _):
        raise ValueError("Predictions is a read-only property. Call fit_predict to fit standard curves and "
                         "popualte predictions.")

    def _prepare_standards_data(self,
                                analyte: str,
                                transform: str or None = None):
        """
        Prepare the standard concentration data for a given analyte using the raw data.

        Parameters
        ----------
        analyte: str
        transform: str, optional

        Returns
        -------
        Pandas.DataFrame
        """
        standards = self.raw[self.raw.Sample.isin(self.standards)][["Sample", analyte]].copy()
        standard_concs = self.concentrations[self.concentrations.analyte == analyte].copy()
        standard_concs = standard_concs[self.standards].melt(var_name="Sample", value_name="conc")
        standards = standards.merge(standard_concs, on="Sample")
        if transform:
            standards = apply_transform(standards,
                                        transform_method=transform,
                                        features_to_transform=[analyte, "conc"])
        return standards

    def _fit(self,
             func: callable,
             transform: str or None,
             analyte: str,
             starting_params: dict or None = None,
             **kwargs):
        """
        Fit the standard curve function for a single analyte.

        Parameters
        ----------
        func: callable
        transform: str
        analyte: str
        kwargs:
            Additional keyword arguments to pass to scipy.optimise.curve_fit function

        Returns
        -------
        None
        """
        data = self._prepare_standards_data(analyte=analyte, transform=transform)
        # model = Model(func)
        params, covar_matrix = curve_fit(func, data[analyte].values, data["conc"].values, **kwargs)
        err = residuals(func,
                        x=data[analyte].values,
                        y=data["conc"].values,
                        params=params)
        self.standard_curves[analyte] = {"params": params,
                                         "transform": transform,
                                         "function": func,
                                         "residuals": err,
                                         "r_squared": r_squared(err=err, y=data["conc"].values),
                                         "sigma": np.sqrt(np.diagonal(covar_matrix))}

    def fit(self,
            func: callable or str,
            transform: str or None = None,
            analyte: str or None = None,
            **kwargs):
        """
        Fit a function to generate one ore more standard curves. The standard curves
        are generated using SciPy's curve_fit function, which uses least squares regression.
        The results of each standard curve, one for each analyte, are stored in a dictionary
        in the standard_curves attribute. The curves are stored like so:

        {"params": the estimated parameters for optimal fit,
         "transform": transformation applied prior to fitting,
         "function": function used to generate standard curve,
         "residuals": residuals for fitted function,
         "r_squared": r-squared value for fitted function",
         "sigma": rough approximation of the error for the estimated parameters}

        If the function fails to estimate the optimal parameters, indicated by a RuntimeError,
        you can try modifying the starting parameters. By default the parameters from
        CytoPy.flow.ext_tools.INIT_PARAMS are used but can be overwritten by passing an array of
        values as the 'p0' parameter in kwargs. The positional order of this array should match the order of
        parameters for the chosen function.

        Parameters
        ----------
        func: str or callable
            One of the following string values should be provided: "linear", "poly2", "poly3", "4pl" or "5pl"
            (where 4pl and 5pl are four and five parameter logistic functions, respectively). Alternatively,
            a custom function can be provided
        transform: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the dependent variable (standard measurements) and the independent variable (standard concentrations)
            prior to fitting the function
        analyte: str, optional
            The analyte to calculate the standard curve for. If not given, all analytes will be fitted in sequence.
        kwargs:
            Additional keyword arguments to pass to scipy.optimise.curve_fit function

        Returns
        -------
        None
        """
        if isinstance(func, str):
            func = _fitting_functions(func)
            kwargs["p0"] = kwargs.get("p0") or INIT_PARAMS.get(func)
            kwargs["maxfev"] = kwargs.get("maxfev") or 10000
        if isinstance(analyte, str):
            self._fit(func, transform, analyte, **kwargs)
        else:
            for analyte in progress_bar(self.analytes):
                self._fit(func, transform, analyte, **kwargs)

    def _predict(self,
                 analyte: str):
        """
        Under the assumption of an existing standard curve, predict the concentration of the
        desired analyte (which must be present as a variable in raw data) using the standard curve.
        Results are stored as a numpy array in predictions.

        Parameters
        ----------
        analyte: str

        Returns
        -------
        None
        """
        x = self.raw[~self.raw.Sample.isin(self.standards)][[analyte]]
        transform = self.standard_curves[analyte].get("transform")
        if transform:
            x = apply_transform(x, features_to_transform=[analyte], transform_method=transform)[analyte].values
        else:
            x = x[analyte].values
        params = self.standard_curves[analyte].get("params")
        yhat = self.standard_curves[analyte].get("function")(x, *params)
        if np.isnan(yhat).any():
            warn("One or more predicted values are Null; will be replaced with zeros")
        self._predictions[analyte] = np.nan_to_num(yhat)

    @assert_fitted
    def predict(self,
                analyte: str or None = None):
        """
        Under the assumption of an existing standard curve, predict the concentration of the
        desired analyte (which must be present as a variable in raw data) using the standard curve.
        Results are stored as a numpy array in predictions.

        Parameters
        ----------
        analyte

        Returns
        -------

        Raises
        ------
        AssertionError
            If standard curve has not been calculated will raise an error.

        """
        if isinstance(analyte, str):
            self._predict(analyte)
        else:
            for analyte in progress_bar(self.analytes):
                self._predict(analyte)

    def fit_predict(self,
                    func: callable or str,
                    transform: str or None = None,
                    analyte: str or None = None,
                    **kwargs):
        """
        Calculate standard curve for the chosen analyte (see fit method for details) and predict
        (see predict method for details) concentrations. Predictions are stored to predictions attribute.

        Parameters
        ----------
        func: str or callable
            One of the following string values should be provided: "linear", "poly2", "poly3", "4pl" or "5pl"
            (where 4pl and 5pl are four and five parameter logistic functions, respectively). Alternatively,
            a custom function can be provided
        transform: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the dependent variable (standard measurements) and the independent variable (standard concentrations)
            prior to fitting the function
        analyte: str, optional
            The analyte to calculate the standard curve for. If not given, all analytes will be fitted in sequence.
        kwargs:
            Additional keyword arguments to pass to scipy.optimise.curve_fit function

        Returns
        -------
        None
        """
        self.fit(func=func, transform=transform, analyte=analyte, **kwargs)
        self.predict(analyte=analyte)

    @staticmethod
    def _inverse_log(analyte: str,
                     xx: np.ndarray,
                     yhat: np.ndarray,
                     data: pd.DataFrame,
                     lower_bound: np.ndarray,
                     upper_bound: np.ndarray,
                     transform: str):
        """
        Given an analyte that has had some log transform applied apply the inverse to
        return values on a linear scale

        Parameters
        ----------
        analyte: str
        xx: Numpy.Array

        yhat
        data
        lower_bound
        upper_bound
        transform

        Returns
        -------

        """
        inverse_log = {"log": lambda x: np.e ** x,
                       "log2": lambda x: 2 ** x,
                       "log10": lambda x: 10 ** x}
        xx = list(map(inverse_log.get(transform), xx))
        yhat = list(map(inverse_log.get(transform), yhat))
        lower_bound = list(map(inverse_log.get(transform), lower_bound))
        upper_bound = list(map(inverse_log.get(transform), upper_bound))
        data[analyte] = data[analyte].apply(inverse_log.get(transform))
        data["conc"] = data["conc"].apply(inverse_log.get(transform))
        return xx, yhat, data, lower_bound, upper_bound

    @assert_fitted
    def plot_standard_curve(self,
                            analyte: str,
                            ax: plt.Axes or None = None):
        ax = ax or plt.subplots(figsize=(8, 8))[1]
        if analyte not in self._predictions.keys():
            self.predict(analyte=analyte)

        data = self._prepare_standards_data(analyte=analyte,
                                            transform=self.standard_curves.get(analyte).get("transform"))
        xx = np.linspace(data[analyte].min() - (data[analyte].min() * 0.01),
                         data[analyte].max() + (data[analyte].max() * 0.01))
        params = self.standard_curves.get(analyte).get("params")
        yhat = self.standard_curves.get(analyte).get("function")(xx, *params)
        sigma = self.standard_curves.get(analyte).get("sigma")
        upper_bound = self.standard_curves.get(analyte).get("function")(xx, *(params + sigma))
        lower_bound = self.standard_curves.get(analyte).get("function")(xx, *(params - sigma))
        applied_transform = self.standard_curves.get(analyte).get("transform")
        if applied_transform in ["log", "log2", "log10"]:
            xx, yhat, data, lower_bound, upper_bound = self._inverse_log(analyte=analyte,
                                                                         xx=xx,
                                                                         yhat=yhat,
                                                                         data=data,
                                                                         upper_bound=upper_bound,
                                                                         lower_bound=lower_bound,
                                                                         transform=applied_transform)
        ax.scatter(data[analyte], data["conc"], facecolor="white", edgecolor="k", s=10, alpha=1)
        ax.plot(xx, yhat, "black")
        ax.fill_between(xx, lower_bound, upper_bound, color="black", alpha=0.15)
        base = {"log": np.e,
                "log2": 2,
                "log10": 10}
        if applied_transform in ["log", "log2", "log10"]:
            ax.set_xscale("log", basex=base.get(applied_transform))
            ax.set_yscale("log", basey=base.get(applied_transform))
        ax.set_xlabel("Response")
        ax.set_ylabel("Concentration")
        return ax

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

    @assert_fitted
    def heatmap(self):
        pass
