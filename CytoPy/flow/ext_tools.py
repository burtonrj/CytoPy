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

INVERSE_LOG = {"log": lambda x: np.e ** x,
               "log2": lambda x: 2 ** x,
               "log10": lambda x: 10 ** x}
BASE = {"log": np.e,
        "log2": 2,
        "log10": 10}


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
    Five parameter logistic function.

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


class Logit(Model, ABC):
    """
    Logistic curve fitting model with either four or five parameters. Inherits from lmfit.Model.
    When constructed user should specify whether to use four or five parameter fit by specifying
    True or False for the 'five_parameter_fit' parameter. If 'use_default_params' is set to True, then
    the following parameter hints will be set (name: value [lower bound, upper bound]):
    * min_: 0 [-1e4,  1e4]
    * max_: 1e4 [1, 1e6]
    * inflection_point: 0.5 [1e-4, 1]
    * coef: 1 [1e-4, 1e4]
    * asymmetry_factor (only for five parameter fit): 0.5 [1e-4, 1]

    If you feel that these default starting parameters do not reflect your dataset, then set "use_default_params"
    to False and set parameter hints as per lmfit docs:
    https://lmfit.github.io/lmfit-py/model.html#initializing-values-with-model-make-params
    """

    def __init__(self,
                 five_parameter_fit: bool = True,
                 use_default_params: bool = True,
                 **kws):
        func = five_param_logit
        if not five_parameter_fit:
            func = four_param_logit
        super().__init__(func, **kws)
        if use_default_params:
            self.set_param_hint(name="min_",
                                value=0,
                                min=-1e4,
                                max=1e4)
            self.set_param_hint(name="max_",
                                value=1e4,
                                min=1,
                                max=1e6)
            self.set_param_hint(name="inflection_point",
                                value=0.5,
                                min=1e-4,
                                max=1)
            self.set_param_hint(name="coef",
                                value=1,
                                min=1e-4,
                                max=1e4)
            if five_parameter_fit:
                self.set_param_hint(name="asymmetry_factor",
                                    value=0.5,
                                    min=1e-4,
                                    max=1)


def default_models(model: str,
                   model_init_kwargs: dict or None = None):
    """
    Generates a default Model object of either Linear, Polynomial, Quadratic or Logistic function.

    Parameters
    ----------
    model: str
        Should be one of linear', 'quad', 'poly', or 'logit'
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
    if model == "logit":
        return Logit(use_default_params=True, **model_init_kwargs)
    raise ValueError("Invalid model, must be one of: 'linear', 'quad', 'poly', or 'logit'")


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


class AssayTools:
    """
    Tools for analysis of plate based assay data such as ELISAs and Luminex assays.
    Calculate standard curves, predict concentrations, transform data, as well as
    access to various plotting functions and statistical tests.

    AssayTools makes heavy use of the lmfit library for fitting curves to data. We recommend the user
    consults their documentation for more information and troubleshooting: https://lmfit.github.io/

    The results of fitted curves are stored in the attribute 'predictions' as a dictionary where each key
    is the analyte name and the value a nested dictionary containing the transform applied and the ModelResult
    object.

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

    @property
    def predictions_linear(self):
        x = self.predictions
        for analyte in self._predictions.keys():
            transform = self.standard_curves.get(analyte).get("transform")
            if transform in ["log", "log2", "log10"]:
                x[analyte] = x[analyte].apply(INVERSE_LOG.get(transform))
            elif transform is not None:
                warn(f"Transform {transform} applied to analyte {analyte} does not have a supported inverse function")
        return x

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
             model: Model,
             transform: str or None,
             analyte: str,
             params: dict or None = None,
             guess_start_params: bool = True,
             **kwargs):
        """
        Fit the standard curve function for a single analyte.

        Parameters
        ----------
        model: Model
        transform: str
        analyte: str
        params: dict, optional
            Optional starting parameters; will overwrite defaults
        kwargs:
            Additional keyword arguments to pass to Model.fit call

        Returns
        -------
        None
        """
        params = params or {}
        standards = self._prepare_standards_data(analyte=analyte, transform=transform)
        if guess_start_params:
            try:
                params = model.guess(data=standards["conc"].values,
                                     x=standards[analyte].values)
            except NotImplementedError:
                params = model.make_params(**params)
        else:
            params = model.make_params(**params)
        self.standard_curves[analyte] = {"transform": transform,
                                         "model_result": model.fit(standards["conc"].values,
                                                                   params=params,
                                                                   x=standards[analyte].values,
                                                                   **kwargs)}

    def fit(self,
            model: Model or str,
            transform: str or None = None,
            analyte: str or None = None,
            starting_params: dict or None = None,
            guess_start_params: bool = True,
            model_init_kwargs: dict or None = None,
            **kwargs):
        """
        Fit a function to generate one or more standard curves. The standard curves
        are generated using the lmfit library (https://lmfit.github.io/), which uses least squares regression.
        A Model object should be provided or a string value which will load a default model for convenience.
        If starting_params is provided, then the specified starting parameters will be used for the initial fit,
        otherwise defaults are used.
        The resulting fit generates a ModelResult object which is stored in the standard_curves attribute, which is a
        dictionary where the key corresponds to the analyte and the value a nested dictionary like so:

        {"transform": transformation applied prior to fitting,
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
            provided: "linear", "quad", "poly" or "logit", generating a LinearModel, QuadraticModel, PolynomialModel
            or "Logit" model. If  "logit" is used, then this will default to a five parameter logistic fit with
            default starting parameters (see CytoPy.flow.ext_tools.Logit for details).
        transform: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the dependent variable (standard measurements) and the independent variable (standard concentrations)
            prior to fitting the function
        analyte: str, optional
            The analyte to calculate the standard curve for. If not given, all analytes will be fitted in sequence.
        starting_params: dict, optional
            Staring parameters for chosen function. If not provided, default starting values will be used
            depending on the given model. If parameters hints have been defined this will overwrite those values.
        guess_start_params: bool (default=True)
            If True, will attempt to guess the optimal starting parameters
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
            model = default_models(model=model, model_init_kwargs=model_init_kwargs)
        if isinstance(analyte, str):
            self._fit(model, transform, analyte, starting_params, guess_start_params, **kwargs)
        else:
            for analyte in progress_bar(self.analytes):
                self._fit(model, transform, analyte, starting_params, guess_start_params, **kwargs)

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
        yhat = self.standard_curves[analyte].get("model_result").eval(x=x)
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
                    model: Model or str,
                    transform: str or None = None,
                    analyte: str or None = None,
                    starting_params: dict or None = None,
                    guess_start_params: bool = True,
                    model_init_kwargs: dict or None = None,
                    **kwargs):
        """
        Calculate standard curve for the chosen analyte (see fit method for details) and predict
        (see predict method for details) concentrations. Predictions are stored to predictions attribute.

        Parameters
        ----------
        model: Model or str
            A valid lmfit.Model object. Alternatively, for convenience, one of the following string values can be
            provided: "linear", "quad", "poly" or "logit", generating a LinearModel, QuadraticModel, PolynomialModel
            or "Logit" model. If  "logit" is used, then this will default to a five parameter logistic fit with
            default starting parameters (see CytoPy.flow.ext_tools.Logit for details).
        transform: str, optional
            If provided, should be a valid transform as supported by CytoPy.flow.transforms and will be applied
            to the dependent variable (standard measurements) and the independent variable (standard concentrations)
            prior to fitting the function
        analyte: str, optional
            The analyte to calculate the standard curve for. If not given, all analytes will be fitted in sequence.
        starting_params: dict, optional
            Staring parameters for chosen function. If not provided, default starting values will be used
            depending on the given model. If parameters hints have been defined this will overwrite those values.
        guess_start_params: bool (default=True)
            If True, will attempt to guess the optimal starting parameters
        model_init_kwargs: dict, optional
            Optional additional keyword arguments to pass if 'model' is of type String. Default models will be
            initialised with the given parameters.
        kwargs:
            Additional keyword arguments to pass to scipy.optimise.curve_fit function

        Returns
        -------
        None
        """
        self.fit(model=model,
                 transform=transform,
                 analyte=analyte,
                 starting_params=starting_params,
                 guess_start_params=guess_start_params,
                 model_init_kwargs=model_init_kwargs,
                 **kwargs)
        self.predict(analyte=analyte)

    def _inverse_log(self,
                     *args,
                     analyte: str):
        """
        For one or more arrays associated to some given analyte, apply the inverse log function according to the
        base logarithm applied to that analyte when generating its standard curve.

        Parameters
        ----------
        args: List[Array]
            One or more array(s)
        analyte: str
            Analyte in question
        Returns
        -------
        List[Array]
            List of transformed arrays
        """
        applied_transform = self.standard_curves.get(analyte).get("transform")
        if applied_transform in ["log", "log2", "log10"]:
            return [list(map(INVERSE_LOG.get(applied_transform), x)) for x in args]
        return args

    def _overlay_predictions(self,
                             analyte: str,
                             ax: plt.Axes,
                             plot_kwargs: dict or None = None):
        """
        Given the standard curve of an analyte (ax) overlay the predicted values for this analyte
        as scatter points.

        Parameters
        ----------
        analyte: str
        ax: Matplotlib.Axes
        plot_kwargs: dict, optional
            Passed to Axes.scatter call (overwrites defaults)

        Returns
        -------
        Matplotlib.Axes
        """
        plot_kwargs = plot_kwargs or dict(s=25,
                                          color="red",
                                          zorder=3,
                                          marker="x")
        if analyte not in self._predictions.keys():
            self.predict(analyte=analyte)
        x = self.predictions[analyte].values
        yhat = self.standard_curves[analyte].get("model_result").eval(x=x)
        x, yhat = self._inverse_log(x, yhat, analyte=analyte)
        ax.scatter(x, yhat, **plot_kwargs)
        return ax

    @assert_fitted
    def plot_standard_curve(self,
                            analyte: str,
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
        data = self._prepare_standards_data(analyte=analyte,
                                            transform=self.standard_curves.get(analyte).get("transform"))
        xcurve = np.linspace(data[analyte].min() - (data[analyte].min() * 0.01),
                             data[analyte].max() + (data[analyte].max() * 0.01))
        ycurve = self.standard_curves[analyte].get("model_result").eval(x=xcurve)
        xscatter = data[analyte].values
        yscatter = data["conc"].values
        xcurve, ycurve, xscatter, yscatter = self._inverse_log(xcurve, ycurve, xscatter, yscatter, analyte=analyte)
        ax.scatter(xscatter, yscatter, **scatter_kwargs)
        ax.plot(xcurve, ycurve, "black", **line_kwargs)
        if self.standard_curves.get(analyte).get("transform") in ["log", "log2", "log10"]:
            b = BASE.get(self.standard_curves.get(analyte).get("transform"))
            ax.set_xscale("log", basex=b)
            ax.set_yscale("log", basey=b)
        ax.set_xlabel("Response")
        ax.set_ylabel("Concentration")
        if overlay_predictions:
            return self._overlay_predictions(analyte=analyte, ax=ax, plot_kwargs=overlay_kwargs)
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

    @assert_fitted
    def plot_repeat_measures(self,
                             analyte: str,
                             log_axis: bool = True,
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
        log_axis: bool (default=True)
            If True, data is assumed to have had a logarithmic transformation applied and the y-axis will
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
        if log_axis:
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
        if log_axis:
            ax.set_yscale("log", ybase=BASE.get(self.standard_curves.get(analyte).get("transform")))
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
        assert factor in self.raw.columns, "Factor is not a valid variable. You can generate meta variables with the " \
                                           "'load_meta' function"
        assert self.raw[factor].nunique() == 2, "Factor must be binary"
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
                       log_axis: bool = True,
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
        log_axis: bool (default=True)
            If True, data is assumed to have had a logarithmic transformation applied and the y-axis will
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
        assert factor in self.raw.columns, "Factor is not a valid variable. You can generate meta variables with the " \
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
            ax.set_yscale("log", ybase=BASE.get(self.standard_curves.get(analyte).get("transform")))
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
        assert factor in self.raw.columns, "Factor is not a valid variable. You can generate meta variables with the " \
                                           "'load_meta' function"
        assert self.raw[factor].nunique() == 2, "Factor must be binary"
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
        subjects = [subject.safe_search(identity_mapper.get(i, i)) for i in self.raw["Sample"].values]
        meta_var_values = list()
        for s in subjects:
            if s is None:
                meta_var_values.append(None)
            else:
                meta_var_values.append(s[meta_var])
        self.raw[meta_var] = meta_var_values
