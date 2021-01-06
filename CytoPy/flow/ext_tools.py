from ..data import subject
from ..feedback import progress_bar
from .descriptives import box_swarm_plot
from .transform import apply_transform
from warnings import warn
from lmfit.models import LinearModel, QuadraticModel, PolynomialModel
from lmfit import Model
from abc import ABC
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import pingouin
import pandas as pd
import numpy as np

np.seterr(over="raise")

INVERSE_LOG = {"log": lambda x: np.e ** x,
               "log2": lambda x: 2 ** x,
               "log10": lambda x: 10 ** x}


def inverse_log(*args,
                transform: str):
    """
    For one or more previously transformed array's, apply the inverse transform

    Parameters
    ----------
    args: List[Array]
        One or more array(s)
    transform: str
        Method used for transform
    Returns
    -------
    List[Array]
        List of inverse transformed arrays
    """
    if transform in ["log", "log2", "log10"]:
        return [list(map(INVERSE_LOG.get(transform), x)) for x in args]
    return args


def generalised_hill_equation(x: np.ndarray,
                              a: float,
                              d: float,
                              slope: float,
                              log_inflection_point: float,
                              symmetry: float = 1.0):
    """
    Five parameter logistic fit for dose-response curves. If four parameter fit is desired, symmetry should have a 
    value of 1.0.

    Paul G. Gottschalk, John R. Dunn, The five-parameter logistic: A characterization and comparison with the
    four-parameter logistic, Analytical Biochemistry, Volume 343, Issue 1, 2005 https://doi.org/10.1016/j.ab.2005.04.035

    Parameters
    ----------
    x: Numpy.Array
        X-axis variable (i.e. response such as OD or MFI)
    a: float
        Bottom asymptote
    d: float
        Top asymptote
    slope: float
        Steepness of the curve at the inflection point
    log_inflection_point: float
        Inflection point as read on the x-axis (scale equivalent to dose)
    symmetry: float
        Degree of asymmetry

    Returns
    -------
    Numpy.Array
    """

    assert slope > 0, "parameter 'slope' must be greater than 0"
    assert symmetry > 0, "parameter 'symmetry' must be greater than 0"
    x = np.log(x)
    logxb = log_inflection_point + (1/slope) * np.log((2 ** (1 / symmetry)) - 1)
    num = d - a
    denom = (1 + 10 ** ((logxb - x) * slope)) ** symmetry
    return a + (num / denom)


class LogisticCurveFit():
    def __init__(self):
        


class LogisticDoseCurve(Model, ABC):
    """
    Logistic curve fitting model with either four or five parameters using the generalised hill equation.
    Inherits from lmfit.Model. When constructed user should specify whether to use four or five parameter fit by
    specifying True or False for the 'five_parameter_fit' parameter. If False, the symmetry parameter is forced
    to have a value of 1.

    The response (x-axis variable) is log10 transformed. If variability in the concentration variable (y-axis)
    is large at low and high values, consider transforming the response variable prior to calling 'fit'.

    Parameter values are not constrained by default, but it is recommended to constrain parameters using parameter
    hints (https://lmfit.github.io/lmfit-py/model.html#initializing-values-with-model-make-params). Alternatively,
    the user can call the 'set_parameter_bounds' method to set suitable constraints based on the dose data,
    and this is the recommended method for those unfamiliar of this function.
    """

    def __init__(self,
                 five_parameter_fit: bool = True,
                 **kws):
        super().__init__(generalised_hill_equation, **kws)
        self.five_parameter_fit = five_parameter_fit
        if not five_parameter_fit:
            self.set_param_hint(name="symmetry",
                                value=1.0,
                                min=1.0,
                                max=1.0)

    def set_parameter_bounds(self,
                             x: np.ndarray,
                             y: np.ndarray,
                             f: float = 0.5):
        """
        Generate parameters with appropriate limits based on the input response data (x). The parameter bounds
        are chosen as such:
        * a (the bottom asymptote) is bound between the range of the bottom 25th percentile of the concentration (y),
        +/- u * f, where u is the median concentration and f is a float (default=0.1)
        * d (the top asymptote) is bound between the range of the top 25th percentile of the concentration (y),
        +/- u * f, where u is the median response value and f is a float (default=0.1)
        * symmetry (controls the asymmetric behavior of the curve) is bound between 1e-5 and 5 unless the object
        was initialised as a four parameter fit, in which case it's value is fixed to 1.0
        * slope (the steepness of the curve at the inflection point) is bound between 1e-5 and max(y)
        * inflection point is bound within the range of response variable (x)

        Parameters
        ----------
        y: Numpy.Array
            The concentrations measured e.g. standard concentrations (if required, should be transformed prior to
            calling this function)
        x: Numpy.Array
            The response data to fit (e.g. OD or MFI measured for standards).
            Raw values should be provided, log10 transform is applied prior to generating bounds.
        f: float (default=0.1)
            Used to control the bounds of the bottom and top asymptote (higher value will result in larger bounds)

        Returns
        -------
        lmfit.
        """
        x = np.log(x)
        y = np.array(y)

        fa = (np.median(y) * f)
        a_min = np.min(y) - fa
        a_max = np.min(y) + fa

        d_start = np.max(y)
        d_min = np.max(y) - fa
        d_max = np.max(y) + fa

        self.set_param_hint(name="a", value=0, min=a_min, max=a_max)
        self.set_param_hint(name="d", value=d_start, min=d_min, max=d_max)
        self.set_param_hint(name="inflection_point", value=np.median(x), min=np.min(x), max=np.max(x))
        self.set_param_hint(name="slope", value=1, min=1e-5, max=np.max(y))
        if self.five_parameter_fit:
            self.set_param_hint(name="symmetry", value=0.5, min=1e-5, max=5)


def default_models(model: str,
                   model_init_kwargs: dict or None = None):
    """
    Generates a default Model object of either Linear, Polynomial, Quadratic or Logistic function.

    Parameters
    ----------
    model: str
        Should be one of linear', 'quad', 'poly', or 'logistic'
    model_init_kwargs: dict, optional
        Additional keyword arguments passed when constructing Model

    Returns
    -------
    Model

    Raises
    -------
    ValueError
        If model is not one of 'linear', 'quad', 'poly', or 'logit'
    """
    model_init_kwargs = model_init_kwargs or {}
    if model == "linear":
        model = LinearModel(**model_init_kwargs)
        model.set_param_hint(name="intercept", value=1, min=-1e9, max=1e9)
        model.set_param_hint(name="slope", value=0.5, min=1e-9, max=1e9)
        return model
    if model == "quad":
        model = QuadraticModel(**model_init_kwargs)
        model.set_param_hint("a", value=0.5, min=1e-5, max=1e5)
        model.set_param_hint("b", value=0.5, min=1e-5, max=1e5)
        model.set_param_hint("c", value=0.5, min=1e-5, max=1e5)
        return model
    if model == "poly":
        degree = model_init_kwargs.pop("degree", 2)
        model = PolynomialModel(degree=degree, **model_init_kwargs)
        for i in range(degree):
            model.set_param_hint(name=f"c{i}", value=0.5, min=1e-9, max=1e9)
        return model
    if model == "logistic":
        return LogisticDoseCurve(**model_init_kwargs)
    raise ValueError("Invalid model, must be one of: 'linear', 'quad', 'poly', or 'logistic'")


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
            if kwargs.get("analyte") is None:
                return method(*args, **kwargs)
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


class AssayTools:
    """
    Tools for analysis of plate based assay data such as ELISAs and Luminex assays. The assumption here is that
    you have some given response e.g. optical density or MFI and you wish to predict some unknown concentration for
    one or more analytes and one or more samples.

    Calculate standard curves, predict concentrations (dose), transform data, as well as
    access to various plotting functions and statistical tests.

    AssayTools makes heavy use of the lmfit library for fitting curves to data. We recommend the user
    consults their documentation for more information and troubleshooting: https://lmfit.github.io/

    The results of fitted curves are stored in the attribute 'predictions'. If standard concentrations have been
    transformed when generating the standard curves, the output in predictions will be on this transformed scale.
    To access predictions on a linear scale, access the 'predictions_linear' attribute. Note: inverse transformation
    is only supported for log, log2, and log10 transforms, therefore 'predictions_linear' will fail if
    some other unsupported transformation has been applied to the response variable.


    Attributes
    ----------
    response: Pandas.DataFrame
        Raw unaltered assay data. Will also contain any associated meta data if the subject ID is
        provided.
    predictions: Pandas.DataFrame
        Predicted concentrations of analytes using standard curves
    analytes: list
        List of analytes being studied
    concentrations: Pandas.DataFrame
        Concentrations corresponding to each standard control (also known as the 'standard dose')
    standard_curves: dict
        Fitted functions for each analyte. Stores a dictionary for each analyte like:
        {"response_transform": transformation applied to the response variable (y),
         "dose_transform": transformation applied to the dose variable (concentration; x)
         "model_result": lmfit ModelResult object}
    """

    def __init__(self,
                 data: pd.DataFrame,
                 conc: pd.DataFrame,
                 standards: list,
                 background_id: str or None = None,
                 analytes: list or None = None,
                 nan_policy: str or float = 1e-5):
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
        nan_policy: str or float
            How to handle Null/NaN predictions. If a float is provided, null values will be replaced with this
            float value prior to saving the result. If a string is provided, it should either be 'warn' or 'raise',
            where the outcome will be a warning or a ValueError, respectively.
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
        self.response = data
        if background_id:
            self.response = subtract_background(data=data,
                                                background_id=background_id,
                                                analytes=self.analytes)
        if isinstance(nan_policy, str):
            assert nan_policy in ["warn", "raise"], "nan_policy should be a float or a string of value value " \
                                                    "'raise' or 'warn'"
        self.nan_policy = nan_policy

    @property
    def predictions(self):
        x = pd.DataFrame(self._predictions)
        other_vars = [c for c in self.response.columns if c not in self.analytes]
        for c in other_vars:
            x[c] = self.response[~self.response.Sample.isin(self.standards)][c].values
        return x

    @predictions.setter
    def predictions(self, _):
        raise ValueError("Predictions is a read-only property. Call fit_predict to fit standard curves and "
                         "popualte predictions.")

    @property
    def predictions_linear(self):
        x = self.predictions
        for analyte in self._predictions.keys():
            transform = self.standard_curves.get(analyte).get("transform_y")
            if transform in ["log", "log2", "log10"]:
                try:
                    x[analyte] = x[analyte].apply(INVERSE_LOG.get(transform))
                except Exception as e:
                    warn(f"Could not calculate inverse log for {analyte}; {str(e)}.")
            elif transform is not None:
                warn(f"Transform {transform} applied to concentrations of analyte {analyte} "
                     f"does not have a supported inverse function")
        return x

    def transform_response(self,
                           analyte: str,
                           transform: str):
        """
        Apply a transform to the recording sample response (e.g. OD or MFI) and standard response for an analyte
        e.g. min max scaling or z-score normalisation.
        Note, outputs from predicted response using standard curves will be transformed
        to the given scale. If the curve fitting function being applied requires a transformed axis but the user
        desires the output on a linear scale, the transform should be specified when fitting the standard curve
        instead.

        Returns
        -------
        analyte: str
        transform: str
            Available transformations can be found at CytoPy.flow.transforms.apply_transform
        """
        if analyte in self.standard_curves.keys():
            warn("Standard curve has already been fitted for this data. Call fit again for transform to take effect.")
        self.response = apply_transform(self.response, transform_method=transform, features_to_transform=[analyte])

    def _prepare_standards_data(self,
                                analyte: str,
                                transform_x: str or None = None,
                                transform_y: str or None = None):
        """
        Prepare the standard concentration data for a given analyte using the raw data.

        Parameters
        ----------
        analyte: str
        transform_x: str, optional
        transform_y: str, optional

        Returns
        -------
        Pandas.DataFrame
        """
        x = (self.response[self.response.Sample.isin(self.standards)][["Sample", analyte]]
             .copy()
             .rename({analyte: "x"}, axis=1))
        y = (self.concentrations[self.concentrations.analyte == analyte]
             .copy()
             .melt(var_name="Sample", value_name="y"))
        standards = x.merge(y, on="Sample")
        standards["x"] = standards["x"].astype(dtype="float64")
        standards["y"] = standards["y"].astype(dtype="float64")
        if transform_x:
            standards = apply_transform(standards,
                                        transform_method=transform_x,
                                        features_to_transform=["x"])
        if transform_y:
            standards = apply_transform(standards,
                                        transform_method=transform_y,
                                        features_to_transform=["y"])
        return standards

    def _fit(self,
             model: Model,
             transform_x: str or None,
             transform_y: str or None,
             analyte: str,
             params: dict or None = None,
             guess_start_params: bool = True,
             guess_start_params_kwargs: dict or None = None,
             **kwargs):
        """
        Fit the standard curve function for a single analyte.

        Parameters
        ----------
        model: Model
        transform_x: str
        transform_y: str
        analyte: str
        params: dict, optional
            Optional starting parameters and bounds; will overwrite defaults
        guess_start_params: bool (default=True)
            If True, will attempt to guess optimal starting parameters using the mMdels 'guess' method
        guess_start_params_kwargs: dict, optional
            Additional keyword arguments passed to Models 'guess' method
        kwargs:
            Additional keyword arguments to pass to Model.fit call

        Returns
        -------
        None
        """
        guess_start_params_kwargs = guess_start_params_kwargs or {}
        params = params or {}
        standards = self._prepare_standards_data(analyte=analyte,
                                                 transform_x=transform_x,
                                                 transform_y=transform_y)
        if guess_start_params:
            if isinstance(model, LogisticDoseCurve):
                params = model.set_parameter_bounds(x=standards["x"].values,
                                                    y=standards["y"].values,
                                                    **guess_start_params_kwargs)
            else:
                try:
                    params = model.guess(data=standards["y"].values,
                                         **guess_start_params_kwargs)
                except NotImplementedError:
                    params = model.make_params(**params)
        else:
            params = model.make_params(**params)
        self.standard_curves[analyte] = {"transform_x": transform_x,
                                         "transform_y": transform_y,
                                         "model_result": model.fit(standards["y"].values,
                                                                   params=params,
                                                                   x=standards["x"].values,
                                                                   **kwargs)}

    def fit(self,
            model: Model or str,
            transform_x: str or None = None,
            transform_y: str or None = None,
            analyte: str or None = None,
            starting_params: dict or None = None,
            guess_start_params: bool = True,
            guess_start_params_kwargs: dict or None = None,
            model_init_kwargs: dict or None = None,
            **kwargs):
        """
        Fit a function to generate one or more standard curves. The standard curves
        are generated using the lmfit library (https://lmfit.github.io/), which uses least squares regression.
        A Model object should be provided or a string value which will load a default model for convenience.
        If starting_params is provided, then the specified starting parameters will be used for the initial fit,
        otherwise defaults are used (starting_params should follow the conventions for parameter hints set out in the
        limfit documentation).

        The resulting fit generates a ModelResult object which is stored in the standard_curves attribute, which is a
        dictionary where the key corresponds to the analyte and the value a nested dictionary like so:

        {"transform_x": transformation applied to standards response variable (e.g. OD or MFI) prior to fitting,
        "transform_y": transformation applied to standard concentrations prior to fitting,
         "model_result": ModelResult object}

         For more details regarding a ModelResult object, see the lmfit documentation here:
         See https://lmfit.github.io/lmfit-py/model.html?highlight=modelresult#the-modelresult-class

        If the function fails to estimate the optimal parameters, indicated by a RuntimeError,
        you can try modifying the starting parameters or increasing the number of
        evaluations by passing a value for max_nfev (usually 1000-10000 will work) in kwargs.

        Parameters
        ----------
        model: Model or str
            A valid lmfit.Model object. Alternatively, for convenience, one of the following string values can be
            provided: "linear", "quad", "poly" or "logistic", generating a LinearModel, QuadraticModel, PolynomialModel
            or "Logit" model. If  "logit" is used, then this will default to a five parameter logistic fit with
            default starting parameters (see CytoPy.flow.ext_tools.Logit for details).
        transform_y: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the standards captured concentrations
        transform_x: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the standard response variable (measured outputs for standards e.g. OD or MFI)
        analyte: str, optional
            The analyte to calculate the standard curve for. If not given, all analytes will be fitted in sequence.
        starting_params: dict, optional
            Staring parameters for chosen function. If not provided, default starting values will be used
            depending on the given model. If parameters hints have been defined this will overwrite those values.
        guess_start_params: bool (default=True)
            If True, will attempt to guess the optimal starting parameters by calling the chosen Models 'guess' method
        guess_start_params_kwargs: dict, optional
            Additional keyword arguments passed to the Models 'guess' method
        model_init_kwargs: dict, optional
            Optional additional keyword arguments to pass if 'model' is of type String. Default models will be
            initialised with the given parameters.
        kwargs:
            Additional keyword arguments to pass to Model.fit call

        Returns
        -------
        None
        """
        if isinstance(model, str):
            if model == "logistic" and transform_x is not None:
                warn("CytoPy implementation of the generalised hill equation applies a log10 transform to the x "
                     "variable and therefore transform_dose should be 'None' when model == 'logistic'")
                transform_x = None
            model = default_models(model=model, model_init_kwargs=model_init_kwargs)
        if isinstance(analyte, str):
            self._fit(model=model,
                      transform_x=transform_x,
                      transform_y=transform_y,
                      analyte=analyte,
                      params=starting_params,
                      guess_start_params=guess_start_params,
                      guess_start_params_kwargs=guess_start_params_kwargs,
                      **kwargs)
        else:
            for analyte in progress_bar(self.analytes):
                self._fit(model=model,
                          transform_x=transform_x,
                          transform_y=transform_y,
                          analyte=analyte,
                          params=starting_params,
                          guess_start_params=guess_start_params,
                          guess_start_params_kwargs=guess_start_params_kwargs,
                          **kwargs)

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
        sample_response = self.response[~self.response.Sample.isin(self.standards)][[analyte]]
        transform_x = self.standard_curves[analyte].get("transform_x")
        if transform_x:
            sample_response = apply_transform(sample_response,
                                              transform_method=transform_x,
                                              features_to_transform=[analyte])
        yhat = self.standard_curves[analyte].get("model_result").eval(x=sample_response[analyte].values)
        if np.isnan(yhat).any():
            if self.nan_policy == "warn":
                warn("One or more predicted concentrations are Null")
            elif self.nan_policy == "raise":
                raise ValueError("One or more predicted concentrations are Null")
            elif isinstance(self.nan_policy, float):
                warn(f"One or more predicted concentrations are Null; will be replaced with {self.nan_policy}")
                yhat = np.nan_to_num(yhat, nan=self.nan_policy)
            else:
                raise ValueError(f"Invalid nan_policy: {self.nan_policy}, check documentation before creating "
                                 f"AssayTools object, nan_policy must be a float or a string with value 'raise' or "
                                 f"'warn'")
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
                    model: Model or str,
                    transform_x: str or None = None,
                    transform_y: str or None = None,
                    analyte: str or None = None,
                    starting_params: dict or None = None,
                    guess_start_params: bool = True,
                    guess_start_params_kwargs: dict or None = None,
                    model_init_kwargs: dict or None = None,
                    **kwargs):
        """
        Calculate standard curve for the chosen analyte (see fit method for details) and predict
        (see predict method for details) concentrations. Predictions are stored to predictions attribute.

        Parameters
        ----------
        model: Model or str
            A valid lmfit.Model object. Alternatively, for convenience, one of the following string values can be
            provided: "linear", "quad", "poly" or "logistic", generating a LinearModel, QuadraticModel, PolynomialModel
            or "Logit" model. If  "logit" is used, then this will default to a five parameter logistic fit with
            default starting parameters (see CytoPy.flow.ext_tools.Logit for details).
        transform_y: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the standards captured concentrations
        transform_x: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the standard response variable (measured outputs for standards e.g. OD or MFI)
        analyte: str, optional
            The analyte to calculate the standard curve for. If not given, all analytes will be fitted in sequence.
        starting_params: dict, optional
            Staring parameters for chosen function. If not provided, default starting values will be used
            depending on the given model. If parameters hints have been defined this will overwrite those values.
        guess_start_params: bool (default=True)
            If True, will attempt to guess the optimal starting parameters by calling the chosen Models 'guess' method
        guess_start_params_kwargs: dict, optional
            Additional keyword arguments passed to the Models 'guess' method
        model_init_kwargs: dict, optional
            Optional additional keyword arguments to pass if 'model' is of type String. Default models will be
            initialised with the given parameters.
        kwargs:
            Additional keyword arguments to pass to Model.fit call

        Returns
        -------
        None
        """
        self.fit(model=model,
                 transform_x=transform_x,
                 transform_y=transform_y,
                 analyte=analyte,
                 starting_params=starting_params,
                 guess_start_params=guess_start_params,
                 guess_start_params_kwargs=guess_start_params_kwargs,
                 model_init_kwargs=model_init_kwargs,
                 **kwargs)
        self.predict(analyte=analyte)

    def _overlay_predictions(self,
                             analyte: str,
                             ax: plt.Axes,
                             x_log_scale: bool = True,
                             y_log_scale: bool = True,
                             plot_kwargs: dict or None = None):
        """
        Given the standard curve of an analyte (ax) overlay the predicted values for this analyte
        as scatter points. Mutates the given Axes object.

        Parameters
        ----------
        analyte: str
        ax: Matplotlib.Axes
        plot_kwargs: dict, optional
            Passed to Axes.scatter call (overwrites defaults)

        Returns
        -------
        None
        """
        plot_kwargs = plot_kwargs or dict(s=25,
                                          color="red",
                                          zorder=3,
                                          marker="x")

        # Collect response data for analyte and predict concentration
        if analyte not in self._predictions.keys():
            self.predict(analyte=analyte)
        x = apply_transform(self.predictions,
                            features_to_transform=[analyte],
                            transform_method=self.standard_curves.get(analyte).get("transform_x"))[analyte].values
        yhat = self.standard_curves[analyte].get("model_result").eval(x=x)

        # Inverse logarithmic scales prior to plotting
        if x_log_scale:
            x = inverse_log(x, transform=self.standard_curves.get(analyte).get("transform_x"))
        if y_log_scale:
            yhat = inverse_log(yhat, transform=self.standard_curves.get(analyte).get("transform_y"))

        ax.scatter(x, yhat, **plot_kwargs)

    @assert_fitted
    def plot_standard_curve(self,
                            analyte: str,
                            x_log_scale: int or float or None = 10,
                            y_log_scale: int or float or None = 10,
                            xlabel: str = "Response",
                            ylabel: str = "Concentration",
                            overlay_predictions: bool = True,
                            scatter_kwargs: dict or None = None,
                            line_kwargs: dict or None = None,
                            overlay_kwargs: dict or None = None,
                            ax: plt.Axes or None = None):
        """
        Plot the standard curve for an analyte. If a logarithmic transformation was applied during curve fitting
        procedure, data will have an inverse transform applied and then plotted on the appropriate axis (i.e.
        natural log, log2 or log10 axis).

        Parameters
        ----------
        analyte: str
            Analyte to plot
        x_log_scale: int or float, optional
            Base of logarithmic transform to apply to x-axis (set to None, to plot on linear scale)
        y_log_scale: int or float, optional
            Base of logarithmic transform to apply to y-axis (set to None, to plot on linear scale)
        xlabel: str
            X-axis label
        ylabel: str
            Y-axis label
        overlay_predictions: bool (default=True)
            If True, the predicted values for samples measured for this analyte are plotted over
            the standard curve as a scatter plot
        scatter_kwargs: dict, optional
            Additional keyword arguments to pass to call to Axes.scatter when plotting the values
            for standard concentrations.
        line_kwargs: dict, optional
            Additional keyword arguments to pass to call to Axes.plot when plotting the standard curve
        overlay_kwargs: dict, optional
            Additional keyword arguments passed to Axes.scatter when plotting the predicted values for
            sample data
        ax: Matplotlib.Axes, optional
            If provided, used to plot data. Otherwise an axis object will be created with figure size 8x8

        Returns
        -------
        Matplotlib.Axes
        """
        ax = ax or plt.subplots(figsize=(8, 8))[1]
        scatter_kwargs = scatter_kwargs or dict(facecolor="white",
                                                edgecolor="k",
                                                s=30,
                                                alpha=1,
                                                zorder=2)
        line_kwargs = line_kwargs or dict(zorder=1)
        standards = self._prepare_standards_data(analyte=analyte,
                                                 transform_x=self.standard_curves.get(analyte).get("transform_x"),
                                                 transform_y=self.standard_curves.get(analyte).get("transform_y"))

        xcurve = np.linspace(standards["x"].min() - (standards["x"].min() * 0.01),
                             standards["x"].max() + (standards["x"].max() * 0.01))
        ycurve = self.standard_curves[analyte].get("model_result").eval(x=xcurve)
        xscatter = standards["x"].values
        yscatter = standards["y"].values

        # Inverse logarithmic scales prior to plotting
        if x_log_scale is not None:
            xcurve, xscatter = inverse_log(xcurve, xscatter,
                                           transform=self.standard_curves.get(analyte).get("transform_x"))
        if y_log_scale is not None:
            ycurve, yscatter = inverse_log(ycurve, yscatter,
                                           transform=self.standard_curves.get(analyte).get("transform_y"))

        ax.scatter(xscatter, yscatter, **scatter_kwargs)
        ax.plot(xcurve, ycurve, "black", **line_kwargs)

        if overlay_predictions:
            self._overlay_predictions(analyte=analyte,
                                      x_log_scale=x_log_scale is not None,
                                      y_log_scale=y_log_scale is not None,
                                      ax=ax,
                                      plot_kwargs=overlay_kwargs)

        if x_log_scale is not None:
            ax.set_xscale("log", base=x_log_scale)
        if y_log_scale is not None:
            ax.set_yscale("log", base=y_log_scale)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    @assert_fitted
    def coef_var(self,
                 analyte: str,
                 linear_scale: bool = True):
        """
        Returns a Pandas DataFrame of the Coefficient of Variation for the given analyte

        Parameters
        ----------
        analyte: str
        linear_scale: bool (default=True)
            If True, data will be transformed to a linear scale prior to calculating CV

        Returns
        -------
        Pandas.DataFrame
        """
        x = self.predictions
        if linear_scale:
            x = self.predictions_linear
        x = (x.groupby("Sample")[analyte].std() / x.groupby("Sample")[analyte].mean()).reset_index()
        return x.sort_values(analyte)

    def standards_sample_response_kde(self):
        pass

    @assert_fitted
    def plot_repeat_measures(self,
                             analyte: str,
                             log_axis: int or float or None = 10,
                             ax: plt.Axes or None = None,
                             mask: pd.DataFrame or None = None,
                             **kwargs):
        """
        Generates a point plot of repeat measures (where 'Sample' column has the same value). This can be
        useful to see if any samples replicates differ significantly and should be addressed. The repeat values
        are plotted on the y axis with the replicant number (as an integer) on the x-axis.

        Parameters
        ----------
        analyte: str
            Analyte to plot
        log_axis: int or float, optional (default=10)
            Set to None to plot on linear scale. If value is given, concentration predicted from standard curve is
            assumed to have had a logarithmic transformation applied. Provide an int or float value
            to be interpreted as the base of logarithmic transform to apply to y-axis.
            be a log axis
        ax: Matplotlib.Axes, optional
            If provided, used to plot data. Otherwise an axis object will be created with figure size 8x8
        mask: Pandas.DataFrame, optional
            Optional masking DataFrame to filter predictions DataFrame prior to plotting
        kwargs:
            Additional keyword arguments passed to pingouin.plot_paired function

        Returns
        -------
        Matplotlib.Axes
        """
        ax = ax or plt.subplots(figsize=(8, 8))[1]
        if analyte not in self.predictions.columns:
            self.predict(analyte=analyte)
        x = self.predictions
        if mask is not None:
            x = x[mask].copy()
        if log_axis is not None:
            x = self.predictions_linear
        x["Duplication index"] = x.groupby("Sample").cumcount() + 1
        ax = pingouin.plot_paired(data=x,
                                  dv=analyte,
                                  within="Duplication index",
                                  subject="Sample",
                                  boxplot=False,
                                  ax=ax,
                                  colors=['grey', 'grey', 'grey'],
                                  **kwargs)
        if log_axis is not None:
            ax.set_yscale("log", base=log_axis)
        return ax

    @assert_fitted
    def plot_shift(self,
                   analyte: str,
                   factor: str,
                   **kwargs):
        """
        Given some binary factor (a variable assigned using 'load_meta' for example), generate a 'shift plot'
        for a given analyte.
        For more information see: https://pingouin-stats.org/generated/pingouin.plot_shift.html#pingouin.plot_shift

        Parameters
        ----------
        analyte: str
        factor: str
        kwargs:
            Additional keyword arguments passed to pingouin.plot_shift call

        Returns
        -------
        Matplotlib.Figure
        """
        assert factor in self.response.columns, "Factor is not a valid variable. You can generate meta variables with the " \
                                                "'load_meta' function"
        assert self.response[factor].nunique() == 2, "Factor must be binary"
        if analyte not in self.predictions.columns:
            self.predict(analyte=analyte)
        df = self.predictions
        factor_values = df[factor].unique()
        x = df[df[factor] == factor_values[0]][analyte].values
        y = df[df[factor] == factor_values[1]][analyte].values
        return pingouin.plot_shift(x, y, **kwargs)

    @assert_fitted
    def corr_matrix(self,
                    method="spearman",
                    mask: pd.DataFrame or None = None,
                    **kwargs):
        """
        Generates a clustered correlation matrix, where the value in each grid space corresponds to the correlation
        between analytes on the axis.

        Parameters
        ----------
        method: str (default="spearman")
            Method for correlation calculation
            (see https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)
        mask: Pandas.DataFrame, optional
            Pandas.DataFrame used to mask predictions DataFrame prior to generating correlation matrix
        kwargs:
            Additional keyword arguments passed to Seaborn.clustermap call

        Returns
        -------
        Seaborn.ClusterGrid object
        """
        df = self.predictions
        if mask is not None:
            df = df[mask].copy()
        df = df[["Sample"] + [x for x in self._predictions.keys()]]
        corr = df.groupby("Sample").mean().corr(method=method)
        return sns.clustermap(data=corr, **kwargs)

    @assert_fitted
    def plot_box_swarm(self,
                       analyte: str,
                       factor: str,
                       log_axis: int or float or None = 10,
                       ax: plt.Axes or None = None,
                       **kwargs):
        """
        Given some binary factor (a variable assigned using 'load_meta' for example), generate a box and swarm plot
        for an analyte (y-axis), showing how this analyte differs for this factor (x-axis)

        Parameters
        ----------
        analyte: str
            Analyte (y-axis variable)
        factor: str
            Factor (x-axis variable)
        log_axis: int or float, optional (default=10)
            Set to None to plot on linear scale. If value is given, concentration predicted from standard curve is
            assumed to have had a logarithmic transformation applied. Provide an int or float value
            to be interpreted as the base of logarithmic transform to apply to y-axis.
            be a log axis
        ax: Matplotlib.Axes, optional
            If provided, used to plot data. Otherwise an axis object will be created with figure size 8x8
        kwargs:
            Additional keyword arguments passed to CytoPy.flow.descriptives.box_swarm_plot

        Returns
        -------
        Matplotlib.Axes
        """
        ax = ax or plt.subplots(figsize=(8, 8))[1]
        assert factor in self.response.columns, "Factor is not a valid variable. You can generate meta variables with the " \
                                                "'load_meta' function"
        if analyte not in self.predictions.columns:
            self.predict(analyte=analyte)
        df = self.predictions
        if log_axis:
            df = self.predictions_linear
        ax = box_swarm_plot(plot_df=df,
                            x=factor,
                            y=analyte,
                            ax=ax,
                            hue=factor,
                            palette="hls",
                            **kwargs)
        if log_axis:
            ax.set_yscale("log", base=log_axis)
        return ax

    @assert_fitted
    def plot_pval_effsize(self,
                          factor: str,
                          eff_size: str = "CLES",
                          alpha: float = 0.05,
                          correction: str = "holm",
                          interactive: bool = True,
                          ax: plt.Axes or None = None,
                          **plotting_kwargs):
        """
        Given some binary factor (a variable assigned using 'load_meta' for example), test for a significant
        difference between cases for every analyte currently predicted, generating a p-value for each. The test
        used will depend on the properties of the underlying data. If the data is normally distributed, a Welch
        T-test is used, otherwise p-values are generated from a Mann-Whitney U test.

        The negative log p-values (after correction for multiple comparisons) are plotted on the y-axis and
        some chosen effect size on the x-axis; similar to a volcano plot. Significant values will be highlighted
        in blue by default.

        Parameters
        ----------
        factor: str
            Factor; used to separate into groups
        eff_size: str (default="CLES")
            Effect size (x-axis);
            see https://pingouin-stats.org/generated/pingouin.compute_effsize.html#pingouin.compute_effsize
        alpha: float (default=0.05)
            Significance level
        correction: str (default="holm")
            Method used to correct for multiple comparisons;
            see https://pingouin-stats.org/generated/pingouin.multicomp.html#pingouin.multicomp
        interactive: bool (default=True)
            If True, uses Plotly to generate an interactive plot. Hovering over individual data points
            reveals analyte name
        ax: Matplotlib.Axes, optional
            If provided, used to plot data. Otherwise an axis object will be created with figure size 8x8
        plotting_kwargs:
            Additional keyword arguments passed to respective plotting function (Ploly.express.scatter, if
            interactive, else Matplotlib.Axes.scatter)

        Returns
        -------
        Matplotlib.Axes or Plotly.graph_objects.Figure
        """
        stats = self.statistics(factor=factor, eff_size=eff_size, alpha=alpha, correction=correction)
        stats["-log10(p-value)"] = -np.log10(stats["Corrected p-val"])
        if interactive:
            fig = px.scatter(stats,
                             x=eff_size,
                             y="-log10(p-value)",
                             color=f"p<={alpha}",
                             hover_data="Analyte",
                             **plotting_kwargs)
            fig.update_traces(marker=dict(size=12,
                                          line=dict(width=2,
                                                    color="DarkSlateGrey"),
                                          selector=dict(mode="markers")))
            fig.add_hline(y=-np.log10(alpha),
                          line_width=3,
                          line_dash="dash",
                          line_color="blue")
            return fig
        else:
            ax = ax or plt.subplots(figsize=(8, 8))
            sig, nonsig = stats[stats["Reject Null"] is True], stats[stats["Reject Null"] is False]
            ax.scatter(sig[eff_size],
                       sig["-log10(p-value)"],
                       color="#51abdb",
                       edgecolor="#3d3d3d",
                       linewidth=3,
                       s=25,
                       label=f"p<={alpha}")
            ax.scatter(nonsig[eff_size],
                       nonsig["-log10(p-value)"],
                       color="white",
                       edgecolor="#3d3d3d",
                       linewidth=3,
                       s=25,
                       label=f"p>{alpha}")
            ax.axhline(y=-np.log10(alpha), color="#51abdb", linewidth=3, linestyle="dashed")
            ax.legend(bbox_to_anchor=(1, 1.15))
            return ax

    def statistics(self,
                   factor: str,
                   correction: str = "holm",
                   alpha: float = 0.05,
                   eff_size: str = "CLES"):
        """
        Generates a DataFrame of results from statistical inference testing when comparing the values of an
        analyte for some binary factor (a variable assigned using 'load_meta' for example). The test
        used will depend on the properties of the underlying data. If the data is normally distributed, a Welch
        T-test is used, otherwise p-values are generated from a Mann-Whitney U test.

        Parameters
        ----------
        factor: str
            Factor; used to separate into groups
        eff_size: str (default="CLES")
            Effect size (x-axis);
            see https://pingouin-stats.org/generated/pingouin.compute_effsize.html#pingouin.compute_effsize
        alpha: float (default=0.05)
            Significance level
        correction: str (default="holm")
            Method used to correct for multiple comparisons;
            see https://pingouin-stats.org/generated/pingouin.multicomp.html#pingouin.multicomp

        Returns
        -------
        Pandas.DataFrame
        """
        assert factor in self.response.columns, "Factor is not a valid variable. You can generate meta variables with the " \
                                                "'load_meta' function"
        assert self.response[factor].nunique() == 2, "Factor must be binary"
        df = self.predictions
        analytes = list(self._predictions.keys())
        groupings = df.groupby(factor)
        norm = (groupings[analytes]
                .apply(pingouin.normality)
                .reset_index()
                .groupby("level_1")["normal"].all()
                .to_dict())

        stats = list()
        for a in analytes:
            obs = [x[1].values for x in groupings[a]]
            if norm.get(a):
                results = pingouin.ttest(obs[0], obs[1],
                                         paired=False,
                                         tail="two-sided",
                                         correction="auto").reset_index().rename({"index": "stat_test"}, axis=1)
            else:
                results = pingouin.mwu(obs[0], obs[1],
                                       tail="two-sided").reset_index().rename({"index": "stat_test"}, axis=1)
            results[eff_size] = pingouin.compute_effsize(obs[0], obs[1], paired=False, eftype=eff_size)

            results["Analyte"] = a
            stats.append(results)
        stats = pd.concat(stats)
        stats["Reject Null"], stats["Corrected p-val"] = pingouin.multicomp(stats["p-val"].values,
                                                                            alpha=alpha,
                                                                            method=correction)
        return stats

    def load_meta(self,
                  meta_var: str,
                  identity_mapper: dict or None = None):
        """
        Load a meta-variable from associated Subjects. The 'Sample' column provided in the DataFrame
        used to generate this AssayTools object will be assumed to contain the subject ID. If this is not
        the case, a dictionary should be provided in 'identity_mapper', which will match the values in 'Sample'
        to their correct sample ID prior to fetching the meta variable.

        The values for the meta variable will be stored in a new column of the same name. If a Subject document
        cannot be found for a sample or the meta variable is missing, the value will be populated as Null.

        Parameters
        ----------
        meta_var: str
        identity_mapper: dict

        Returns
        -------
        None
        """
        identity_mapper = identity_mapper or {}
        subjects = [subject.safe_search(identity_mapper.get(i, i)) for i in self.response["Sample"].values]
        meta_var_values = list()
        for s in subjects:
            if s is None:
                meta_var_values.append(None)
            else:
                meta_var_values.append(s[meta_var])
        self.response[meta_var] = meta_var_values
